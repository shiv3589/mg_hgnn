"""OULAD heterogeneous graph loader.

Download the Open University Learning Analytics Dataset from:
    https://analyse.kmi.open.ac.uk/open_dataset

Expected files inside *data_dir*:
    studentInfo.csv, studentRegistration.csv, studentAssessment.csv,
    studentVle.csv, courses.csv, assessments.csv, vle.csv
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import Tensor
from torch_geometric.data import HeteroData


class OULADLoader:
    """Builds a PyG :class:`~torch_geometric.data.HeteroData` graph from OULAD.

    The graph schema mirrors the MG-HGNN paper:

    * **Node types**: student, course, instructor (one per module), resource
      (VLE site).
    * **Edge types**: enrolled_in, collaborated_with (co-enrolment proxy),
      accessed, submitted_to.

    Node features are split into three modality tensors per type:

    * ``x``        — structured (tabular) float features.
    * ``text_x``   — 768-d placeholder for BERT embeddings; replace with
                     real embeddings from :mod:`transformers` before training.
    * ``behavior`` — shape ``[N_students, seq_len, 2]`` click-sequence tensor
                     (only populated for student nodes).

    Args:
        data_dir: Path to the directory containing raw OULAD CSV files.
        task: Prediction target — ``"grade"``, ``"dropout"``, or
            ``"engagement"``.
        seq_len: Maximum VLE click-sequence length kept per student (older
            interactions are dropped; shorter sequences are zero-padded).
    """

    _CSV_NAMES = [
        "studentInfo",
        "studentRegistration",
        "studentAssessment",
        "studentVle",
        "courses",
        "assessments",
        "vle",
    ]

    def __init__(self, data_dir: str, task: str = "grade", seq_len: int = 50) -> None:
        self.data_dir = data_dir
        self.task = task
        self.seq_len = seq_len
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scalers: Dict[str, StandardScaler] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[HeteroData, Tensor]:
        """Load OULAD, construct the graph, and return node-level labels.

        Returns:
            data: Populated :class:`~torch_geometric.data.HeteroData` object.
            labels: Float tensor of shape ``[num_students]``.  Semantics
                depend on ``self.task``.
        """
        dfs = self._read_csvs()
        data = HeteroData()

        student_map, student_feats = self._build_students(dfs)
        course_map,  course_feats  = self._build_courses(dfs)
        resource_map, resource_feats = self._build_resources(dfs)
        instructor_map, instructor_feats = self._build_instructors(dfs)

        # Structured features ------------------------------------------------
        data["student"].x    = student_feats
        data["course"].x     = course_feats
        data["resource"].x   = resource_feats
        data["instructor"].x = instructor_feats

        # Textual feature placeholders (replace with real BERT embeddings) ---
        data["student"].text_x    = torch.zeros(len(student_map),    768)
        data["course"].text_x     = torch.zeros(len(course_map),     768)
        data["resource"].text_x   = torch.zeros(len(resource_map),   768)
        data["instructor"].text_x = torch.zeros(len(instructor_map), 768)

        # Behavioral sequences (students only) --------------------------------
        data["student"].behavior = self._build_behavior_sequences(
            dfs, student_map, resource_map
        )

        # Edges ---------------------------------------------------------------
        ei = self._build_edges(dfs, student_map, course_map, resource_map)
        data["student", "enrolled_in",       "course"].edge_index  = ei["enrolled_in"]
        data["student", "collaborated_with", "student"].edge_index = ei["collaborated_with"]
        data["student", "accessed",          "resource"].edge_index = ei["accessed"]
        data["student", "submitted_to",      "course"].edge_index  = ei["submitted_to"]

        labels = self._build_labels(dfs, student_map)
        return data, labels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_csvs(self) -> Dict[str, pd.DataFrame]:
        """Read all OULAD CSV files into a name → DataFrame mapping."""
        return {
            name: pd.read_csv(os.path.join(self.data_dir, f"{name}.csv"))
            for name in self._CSV_NAMES
        }

    def _build_students(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[int, int], Tensor]:
        """Encode student demographic features into a float tensor.

        Returns:
            student_map: Mapping from OULAD ``id_student`` → contiguous index.
            feats: Float tensor of shape ``[N_students, num_features]``.
        """
        df = dfs["studentInfo"].drop_duplicates("id_student").reset_index(drop=True)

        cat_cols = [
            "gender", "region", "highest_education",
            "imd_band", "age_band", "disability",
        ]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
            self._label_encoders[f"student_{col}"] = le

        num_cols = ["num_of_prev_attempts", "studied_credits"]
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))
        self._scalers["student"] = scaler

        student_map = {int(sid): i for i, sid in enumerate(df["id_student"])}
        feats = torch.tensor(df[cat_cols + num_cols].values, dtype=torch.float)
        return student_map, feats

    def _build_courses(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, int], Tensor]:
        """Build course node features from module and presentation metadata.

        Returns:
            course_map: Mapping from course-id string → contiguous index.
            feats: Float tensor of shape ``[N_courses, 3]``.
        """
        df = dfs["courses"].copy()
        df["course_id"] = df["code_module"] + "_" + df["code_presentation"]
        df = df.drop_duplicates("course_id").reset_index(drop=True)

        le_mod  = LabelEncoder().fit(df["code_module"])
        le_pres = LabelEncoder().fit(df["code_presentation"])
        feats = np.stack([
            le_mod.transform(df["code_module"]),
            le_pres.transform(df["code_presentation"]),
            df["module_presentation_length"].fillna(0).values,
        ], axis=1).astype(np.float32)

        course_map = {cid: i for i, cid in enumerate(df["course_id"])}
        return course_map, torch.from_numpy(feats)

    def _build_resources(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[int, int], Tensor]:
        """Build resource node features from VLE site metadata.

        Returns:
            resource_map: Mapping from ``id_site`` (int) → contiguous index.
            feats: Float tensor of shape ``[N_resources, 3]``.
        """
        df = dfs["vle"].drop_duplicates("id_site").reset_index(drop=True)
        le = LabelEncoder()
        activity_enc = le.fit_transform(df["activity_type"].astype(str))
        feats = np.stack([
            activity_enc,
            df["week_from"].fillna(0).values,
            df["week_to"].fillna(0).values,
        ], axis=1).astype(np.float32)

        resource_map = {int(row["id_site"]): i for _, row in df.iterrows()}
        return resource_map, torch.from_numpy(feats)

    def _build_instructors(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, int], Tensor]:
        """Synthesise one instructor node per module (OULAD has no instructor data).

        Returns:
            instructor_map: Mapping from module code → contiguous index.
            feats: Identity matrix of shape ``[N_modules, N_modules]``.
        """
        modules = sorted(dfs["courses"]["code_module"].unique())
        instructor_map = {m: i for i, m in enumerate(modules)}
        feats = torch.eye(len(modules))
        return instructor_map, feats

    def _build_behavior_sequences(
        self,
        dfs: Dict[str, pd.DataFrame],
        student_map: Dict[int, int],
        resource_map: Dict[int, int],
    ) -> Tensor:
        """Build per-student click-sequence tensors from VLE interaction logs.

        Each time-step encodes (resource_index, log1p_click_count).  Sequences
        are right-aligned and zero-padded on the left to ``self.seq_len``.

        Returns:
            Tensor of shape ``[N_students, seq_len, 2]``.
        """
        df = dfs["studentVle"].sort_values(["id_student", "date"])
        sequences = torch.zeros(len(student_map), self.seq_len, 2)

        for sid, grp in df.groupby("id_student"):
            if sid not in student_map:
                continue
            idx = student_map[sid]
            rows = grp[["id_site", "sum_click"]].values[-self.seq_len :]
            for t, (site, clicks) in enumerate(rows):
                res_idx = resource_map.get(int(site), 0)
                sequences[idx, t, 0] = float(res_idx)
                sequences[idx, t, 1] = float(np.log1p(clicks))

        return sequences

    def _build_edges(
        self,
        dfs: Dict[str, pd.DataFrame],
        student_map: Dict[int, int],
        course_map: Dict[str, int],
        resource_map: Dict[int, int],
    ) -> Dict[str, Tensor]:
        """Construct ``edge_index`` tensors for all four relation types.

        Returns:
            Dictionary mapping relation name → ``LongTensor`` of shape
            ``[2, num_edges]``.
        """
        edges: Dict[str, Tensor] = {}

        # enrolled_in: student → course ---------------------------------
        reg = dfs["studentRegistration"].copy()
        reg["course_id"] = reg["code_module"] + "_" + reg["code_presentation"]
        mask = reg["id_student"].isin(student_map) & reg["course_id"].isin(course_map)
        reg = reg[mask]
        src = [student_map[int(s)] for s in reg["id_student"]]
        dst = [course_map[c] for c in reg["course_id"]]
        edges["enrolled_in"] = torch.tensor([src, dst], dtype=torch.long)

        # accessed: student → resource -----------------------------------
        vle = dfs["studentVle"].copy()
        mask = vle["id_student"].isin(student_map) & vle["id_site"].isin(resource_map)
        vle = vle[mask].drop_duplicates(["id_student", "id_site"])
        src = [student_map[int(s)] for s in vle["id_student"]]
        dst = [resource_map[int(r)] for r in vle["id_site"]]
        edges["accessed"] = torch.tensor([src, dst], dtype=torch.long)

        # submitted_to: student → course (via assessment) ----------------
        asmt = dfs["studentAssessment"].merge(
            dfs["assessments"][["id_assessment", "code_module", "code_presentation"]],
            on="id_assessment",
            how="left",
        )
        asmt["course_id"] = asmt["code_module"] + "_" + asmt["code_presentation"]
        mask = asmt["id_student"].isin(student_map) & asmt["course_id"].isin(course_map)
        asmt = asmt[mask].drop_duplicates(["id_student", "course_id"])
        src = [student_map[int(s)] for s in asmt["id_student"]]
        dst = [course_map[c] for c in asmt["course_id"]]
        edges["submitted_to"] = torch.tensor([src, dst], dtype=torch.long)

        # collaborated_with: students co-enrolled in the same presentation
        # Fan-out is capped at 5 peers per student to avoid quadratic blowup.
        reg_full = dfs["studentRegistration"].copy()
        reg_full["course_id"] = reg_full["code_module"] + "_" + reg_full["code_presentation"]
        collab_src, collab_dst = [], []
        for _, grp in reg_full.groupby("course_id"):
            sids = [student_map[int(s)] for s in grp["id_student"] if int(s) in student_map]
            for i, u in enumerate(sids):
                for v in sids[i + 1 : i + 6]:
                    collab_src += [u, v]
                    collab_dst += [v, u]
        edges["collaborated_with"] = torch.tensor(
            [collab_src, collab_dst], dtype=torch.long
        )

        return edges

    def _build_labels(
        self, dfs: Dict[str, pd.DataFrame], student_map: Dict[int, int]
    ) -> Tensor:
        """Return per-student labels aligned with ``student_map`` ordering.

        Returns:
            Float tensor of shape ``[N_students]`` with semantics:

            * ``"grade"``      — mean assessment score in ``[0, 100]``.
            * ``"dropout"``    — binary flag (1 = Withdrawn).
            * ``"engagement"`` — click-count quartile in ``{0, 1, 2, 3}``.
        """
        info = dfs["studentInfo"].drop_duplicates("id_student").copy()
        info = info[info["id_student"].isin(student_map)]
        info = info.sort_values("id_student", key=lambda s: s.map(student_map))

        if self.task == "dropout":
            labels = (info["final_result"] == "Withdrawn").astype(float).values

        elif self.task == "engagement":
            totals = (
                dfs["studentVle"]
                .groupby("id_student")["sum_click"]
                .sum()
                .reindex(info["id_student"])
                .fillna(0)
            )
            labels = pd.qcut(totals, q=4, labels=False).fillna(0).values.astype(float)

        else:  # grade
            scores = (
                dfs["studentAssessment"]
                .groupby("id_student")["score"]
                .mean()
                .reindex(info["id_student"])
                .fillna(0.0)
            )
            labels = scores.values.astype(float)

        return torch.tensor(labels, dtype=torch.float)
