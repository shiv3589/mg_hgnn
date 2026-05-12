"""MOOC-Cube heterogeneous graph loader.

Download the MOOC-Cube dataset from:
    https://github.com/THU-KEG/MOOC-Cube

Expected JSON files inside *data_dir*:
    user.json, course.json, video.json, problem.json,
    user_video_act.json, user_problem_act.json,
    course_concept.json, concept_prerequisite.json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import Tensor
from torch_geometric.data import HeteroData


def _load_json(path: str) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class MOOCCubeLoader:
    """Builds a PyG :class:`~torch_geometric.data.HeteroData` graph from MOOC-Cube.

    Graph schema:

    * **Node types**: student (user), course, instructor (concept), resource
      (video + problem combined).
    * **Edge types**: enrolled_in (user→course), collaborated_with
      (user→user, co-enrolment proxy), accessed (user→resource),
      submitted_to (user→course via problem submission).

    Node features follow the same three-modality convention as OULAD:

    * ``x``        — structured float features.
    * ``text_x``   — 768-d BERT placeholder (replace before training).
    * ``behavior`` — click/watch sequence tensor (students only).

    Args:
        data_dir: Path to the directory containing MOOC-Cube JSON files.
        task: Prediction target — ``"grade"``, ``"dropout"``, or
            ``"engagement"``.
        seq_len: Maximum activity-sequence length per student.
    """

    def __init__(self, data_dir: str, task: str = "grade", seq_len: int = 50) -> None:
        self.data_dir = data_dir
        self.task = task
        self.seq_len = seq_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[HeteroData, Tensor]:
        """Load MOOC-Cube, construct the heterogeneous graph, and return labels.

        Returns:
            data: Populated :class:`~torch_geometric.data.HeteroData` object.
            labels: Float tensor of shape ``[num_students]``.
        """
        raw = self._read_jsons()
        data = HeteroData()

        student_map,    student_feats    = self._build_students(raw)
        course_map,     course_feats     = self._build_courses(raw)
        resource_map,   resource_feats   = self._build_resources(raw)
        instructor_map, instructor_feats = self._build_instructors(raw)

        # Structured features --------------------------------------------
        data["student"].x    = student_feats
        data["course"].x     = course_feats
        data["resource"].x   = resource_feats
        data["instructor"].x = instructor_feats

        # Text placeholders (swap for real BERT embeddings) --------------
        data["student"].text_x    = torch.zeros(len(student_map),    768)
        data["course"].text_x     = torch.zeros(len(course_map),     768)
        data["resource"].text_x   = torch.zeros(len(resource_map),   768)
        data["instructor"].text_x = torch.zeros(len(instructor_map), 768)

        # Behavioral sequences -------------------------------------------
        data["student"].behavior = self._build_behavior_sequences(
            raw, student_map, resource_map
        )

        # Edges ----------------------------------------------------------
        ei = self._build_edges(raw, student_map, course_map, resource_map)
        data["student",  "enrolled_in",       "course"].edge_index  = ei["enrolled_in"]
        data["student",  "collaborated_with", "student"].edge_index = ei["collaborated_with"]
        data["student",  "accessed",          "resource"].edge_index = ei["accessed"]
        data["student",  "submitted_to",      "course"].edge_index  = ei["submitted_to"]

        labels = self._build_labels(raw, student_map, course_map)
        return data, labels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_jsons(self) -> Dict[str, Any]:
        """Read all MOOC-Cube JSON files into a keyed dictionary."""
        names = [
            "user", "course", "video", "problem",
            "user_video_act", "user_problem_act",
            "course_concept", "concept_prerequisite",
        ]
        raw: Dict[str, Any] = {}
        for name in names:
            fpath = os.path.join(self.data_dir, f"{name}.json")
            if os.path.exists(fpath):
                raw[name] = _load_json(fpath)
            else:
                raw[name] = []
        return raw

    def _build_students(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dict[str, int], Tensor]:
        """Encode user records into a structured feature tensor.

        MOOC-Cube user records may contain fields such as ``user_id``,
        ``gender``, ``enroll_count``.  Missing fields default to 0.

        Returns:
            student_map: ``user_id`` → contiguous index.
            feats: Float tensor of shape ``[N_students, 2]``.
        """
        users: List[Dict] = raw.get("user", [])
        if not users:
            raise FileNotFoundError("user.json is empty or missing.")

        student_map = {str(u["user_id"]): i for i, u in enumerate(users)}
        gender_le = LabelEncoder().fit(
            [str(u.get("gender", "unknown")) for u in users]
        )
        feats = np.array([
            [
                gender_le.transform([str(u.get("gender", "unknown"))])[0],
                float(u.get("enroll_count", 0)),
            ]
            for u in users
        ], dtype=np.float32)
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats)
        return student_map, torch.from_numpy(feats)

    def _build_courses(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dict[str, int], Tensor]:
        """Encode course records into a structured feature tensor.

        Returns:
            course_map: ``course_id`` → contiguous index.
            feats: Float tensor of shape ``[N_courses, 2]``.
        """
        courses: List[Dict] = raw.get("course", [])
        if not courses:
            raise FileNotFoundError("course.json is empty or missing.")

        course_map = {str(c["course_id"]): i for i, c in enumerate(courses)}
        feats = np.array([
            [
                float(c.get("video_count", 0)),
                float(c.get("problem_count", 0)),
            ]
            for c in courses
        ], dtype=np.float32)
        return course_map, torch.from_numpy(feats)

    def _build_resources(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dict[str, int], Tensor]:
        """Combine video and problem records into a single resource node set.

        Returns:
            resource_map: ``"v_{id}"`` / ``"p_{id}"`` → contiguous index.
            feats: Float tensor of shape ``[N_resources, 2]``
                   (resource_type_one_hot dim=0: 0=video, 1=problem;
                    dim=1: duration / difficulty).
        """
        rows: List[Tuple[float, float]] = []
        resource_map: Dict[str, int] = {}

        for v in raw.get("video", []):
            key = f"v_{v['video_id']}"
            resource_map[key] = len(resource_map)
            rows.append((0.0, float(v.get("duration", 0))))

        for p in raw.get("problem", []):
            key = f"p_{p['problem_id']}"
            resource_map[key] = len(resource_map)
            rows.append((1.0, float(p.get("difficulty", 0.5))))

        feats = torch.tensor(rows, dtype=torch.float) if rows else torch.zeros(1, 2)
        return resource_map, feats

    def _build_instructors(
        self, raw: Dict[str, Any]
    ) -> Tuple[Dict[str, int], Tensor]:
        """Use knowledge concepts as instructor-proxy nodes.

        Returns:
            instructor_map: concept string → contiguous index.
            feats: Float tensor of shape ``[N_concepts, 1]``.
        """
        concepts: List[str] = []
        for entry in raw.get("course_concept", []):
            for c in entry.get("concepts", []):
                if c not in concepts:
                    concepts.append(c)
        if not concepts:
            concepts = ["placeholder"]
        instructor_map = {c: i for i, c in enumerate(concepts)}
        feats = torch.ones(len(concepts), 1)
        return instructor_map, feats

    def _build_behavior_sequences(
        self,
        raw: Dict[str, Any],
        student_map: Dict[str, int],
        resource_map: Dict[str, int],
    ) -> Tensor:
        """Build per-student video-watch and problem-attempt sequence tensors.

        Each time-step encodes ``(resource_index, interaction_score)``.

        Returns:
            Tensor of shape ``[N_students, seq_len, 2]``.
        """
        sequences = torch.zeros(len(student_map), self.seq_len, 2)

        combined: Dict[str, List[Tuple[str, float]]] = {k: [] for k in student_map}

        for act in raw.get("user_video_act", []):
            uid = str(act.get("user_id", ""))
            vid = f"v_{act.get('video_id', '')}"
            if uid in combined and vid in resource_map:
                combined[uid].append((vid, float(act.get("watch_ratio", 1.0))))

        for act in raw.get("user_problem_act", []):
            uid = str(act.get("user_id", ""))
            pid = f"p_{act.get('problem_id', '')}"
            if uid in combined and pid in resource_map:
                combined[uid].append((pid, float(act.get("score", 0.5))))

        for uid, seq in combined.items():
            i = student_map[uid]
            for t, (res_key, score) in enumerate(seq[-self.seq_len :]):
                sequences[i, t, 0] = float(resource_map[res_key])
                sequences[i, t, 1] = score

        return sequences

    def _build_edges(
        self,
        raw: Dict[str, Any],
        student_map: Dict[str, int],
        course_map: Dict[str, int],
        resource_map: Dict[str, int],
    ) -> Dict[str, Tensor]:
        """Construct edge_index tensors for all four relation types.

        Returns:
            Mapping from relation name → ``LongTensor[2, num_edges]``.
        """
        edges: Dict[str, Tensor] = {}

        # enrolled_in: user → course (derived from user_video_act via course) --
        enroll_src, enroll_dst = [], []
        for c in raw.get("course", []):
            cid = str(c["course_id"])
            if cid not in course_map:
                continue
            for uid in c.get("students", []):
                uid_s = str(uid)
                if uid_s in student_map:
                    enroll_src.append(student_map[uid_s])
                    enroll_dst.append(course_map[cid])
        edges["enrolled_in"] = torch.tensor(
            [enroll_src, enroll_dst], dtype=torch.long
        ) if enroll_src else torch.zeros(2, 0, dtype=torch.long)

        # accessed: user → resource (videos + problems) ----------------------
        acc_src, acc_dst = [], []
        for act in raw.get("user_video_act", []):
            uid, vid = str(act.get("user_id", "")), f"v_{act.get('video_id', '')}"
            if uid in student_map and vid in resource_map:
                acc_src.append(student_map[uid])
                acc_dst.append(resource_map[vid])
        for act in raw.get("user_problem_act", []):
            uid, pid = str(act.get("user_id", "")), f"p_{act.get('problem_id', '')}"
            if uid in student_map and pid in resource_map:
                acc_src.append(student_map[uid])
                acc_dst.append(resource_map[pid])
        # Deduplicate
        seen: set = set()
        dedup_src, dedup_dst = [], []
        for s, d in zip(acc_src, acc_dst):
            if (s, d) not in seen:
                seen.add((s, d))
                dedup_src.append(s)
                dedup_dst.append(d)
        edges["accessed"] = torch.tensor(
            [dedup_src, dedup_dst], dtype=torch.long
        ) if dedup_src else torch.zeros(2, 0, dtype=torch.long)

        # submitted_to: user → course (via problem submission) ---------------
        sub_src, sub_dst = [], []
        for act in raw.get("user_problem_act", []):
            uid = str(act.get("user_id", ""))
            cid = str(act.get("course_id", ""))
            if uid in student_map and cid in course_map:
                sub_src.append(student_map[uid])
                sub_dst.append(course_map[cid])
        edges["submitted_to"] = torch.tensor(
            [sub_src, sub_dst], dtype=torch.long
        ) if sub_src else torch.zeros(2, 0, dtype=torch.long)

        # collaborated_with: co-enrolled students (capped fan-out = 5) --------
        collab_src, collab_dst = [], []
        for c in raw.get("course", []):
            sids = [
                student_map[str(u)]
                for u in c.get("students", [])
                if str(u) in student_map
            ]
            for i, u in enumerate(sids):
                for v in sids[i + 1 : i + 6]:
                    collab_src += [u, v]
                    collab_dst += [v, u]
        edges["collaborated_with"] = torch.tensor(
            [collab_src, collab_dst], dtype=torch.long
        ) if collab_src else torch.zeros(2, 0, dtype=torch.long)

        return edges

    def _build_labels(
        self,
        raw: Dict[str, Any],
        student_map: Dict[str, int],
        course_map: Dict[str, int],
    ) -> Tensor:
        """Return per-student labels for the configured task.

        Returns:
            Float tensor of shape ``[N_students]``.
        """
        n = len(student_map)
        labels = np.zeros(n, dtype=np.float32)

        if self.task == "grade":
            score_sum = np.zeros(n)
            score_cnt = np.zeros(n)
            for act in raw.get("user_problem_act", []):
                uid = str(act.get("user_id", ""))
                if uid in student_map and "score" in act:
                    i = student_map[uid]
                    score_sum[i] += float(act["score"])
                    score_cnt[i] += 1
            mask = score_cnt > 0
            labels[mask] = score_sum[mask] / score_cnt[mask]

        elif self.task == "dropout":
            active = set()
            for act in raw.get("user_video_act", []):
                active.add(str(act.get("user_id", "")))
            for act in raw.get("user_problem_act", []):
                active.add(str(act.get("user_id", "")))
            for uid, i in student_map.items():
                labels[i] = 0.0 if uid in active else 1.0

        else:  # engagement
            total = np.zeros(n)
            for act in raw.get("user_video_act", []):
                uid = str(act.get("user_id", ""))
                if uid in student_map:
                    total[student_map[uid]] += float(act.get("watch_ratio", 0))
            quartiles = np.percentile(total, [25, 50, 75])
            labels = np.digitize(total, quartiles).astype(np.float32)

        return torch.from_numpy(labels)
