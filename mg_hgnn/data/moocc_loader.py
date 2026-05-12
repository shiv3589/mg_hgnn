# Download from: https://github.com/THUwangcy/MOOCCube
# Or: https://huggingface.co/datasets/mooccube
# Place extracted folder at: data/raw/moocc/
# Expected files:
#   entities/user.json           — student info
#   entities/course.json         — course metadata (has text!)
#   entities/video.json          — resource info (has text!)
#   relations/user-video.json    — access behavior
#   relations/user-course.json   — enrollment
#   relations/user-problem.json  — problem attempts

import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer

from config import Config

# ---------------------------------------------------------------
# Files that must exist before loading
# ---------------------------------------------------------------
_REQUIRED = {
    "entities/user.json":          "student demographics / activity counts",
    "entities/course.json":        "course metadata with text descriptions",
    "entities/video.json":         "video resource metadata",
    "relations/user-video.json":   "student–video watch events",
    "relations/user-course.json":  "student–course enrollments",
    "relations/user-problem.json": "student–problem attempt records",
}


class MOOCCubeLoader:

    def __init__(self, config: Config, raw_dir: str = "data/raw/moocc"):
        self.cfg = config
        self.raw = pathlib.Path(raw_dir)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def check_files(self) -> bool:
        """
        Validate that all required JSON files exist.
        Prints a clear error and download instructions if any are missing.
        Returns True if all files are present, False otherwise.
        """
        missing = [rel for rel in _REQUIRED if not (self.raw / rel).exists()]
        if not missing:
            print("MOOCCube files OK — all required files present.")
            return True

        W = 62
        print(f"\n{'!'*W}")
        print("  MOOCCube data files are missing.")
        print(f"  Expected root: {self.raw.resolve()}")
        print(f"\n  Missing files:")
        for rel in missing:
            print(f"    {rel}  ({_REQUIRED[rel]})")
        print(f"\n  Download instructions:")
        print("    Option 1 — GitHub:")
        print("      git clone https://github.com/THUwangcy/MOOCCube")
        print(f"      cp -r MOOCCube/data/* {self.raw}/")
        print("    Option 2 — Hugging Face:")
        print("      pip install huggingface_hub")
        print("      python -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('mooccube', repo_type='dataset', "
              f"local_dir='{self.raw}')\"")
        print(f"{'!'*W}\n")
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_json(self, rel_path: str):
        with open(self.raw / rel_path) as fh:
            return json.load(fh)

    def _tokenize(self, texts: List[str],
                  max_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = AutoTokenizer.from_pretrained(self.cfg.text_model_name)
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load(self) -> Tuple[HeteroData, Dict]:
        """
        Returns (data, meta) with the same schema as OULADLoader so
        train.py can use either loader without modification.
        """
        if not self.check_files():
            raise FileNotFoundError(
                f"Required MOOCCube files missing under {self.raw}. "
                "See instructions above."
            )

        print("Loading MOOCCube JSON files...")
        raw_users    = self._load_json("entities/user.json")
        raw_courses  = self._load_json("entities/course.json")
        raw_videos   = self._load_json("entities/video.json")
        rel_uv       = self._load_json("relations/user-video.json")
        rel_uc       = self._load_json("relations/user-course.json")
        rel_up       = self._load_json("relations/user-problem.json")

        # ── 1. Students ───────────────────────────────────────────────
        # user.json records: {"user_id": str, "enroll_count": int,
        #                     "problem_count": int, "video_count": int}
        student_ids = [u["user_id"] for u in raw_users]
        student_map: Dict[str, int] = {s: i for i, s in enumerate(student_ids)}
        N_s = len(student_ids)
        print(f"  Students:  {N_s:,}")

        enroll_cnt  = np.array([u.get("enroll_count",  0) for u in raw_users], dtype=np.float32)
        problem_cnt = np.array([u.get("problem_count", 0) for u in raw_users], dtype=np.float32)
        video_cnt   = np.array([u.get("video_count",   0) for u in raw_users], dtype=np.float32)

        raw_struct = np.stack([
            np.log1p(enroll_cnt),
            np.log1p(problem_cnt),
            np.log1p(video_cnt),
        ], axis=1)  # (N_s, 3)
        raw_struct = StandardScaler().fit_transform(raw_struct).astype(np.float32)

        x_struct = np.zeros((N_s, self.cfg.structured_input_dim), dtype=np.float32)
        x_struct[:, :raw_struct.shape[1]] = raw_struct

        # ── 2. Courses (real text) ────────────────────────────────────
        # course.json records: {"course_id": str, "name": str, "about": str, ...}
        course_ids  = [c["course_id"] for c in raw_courses]
        course_map: Dict[str, int] = {c: i for i, c in enumerate(course_ids)}
        N_c = len(course_ids)

        # Tokenize 'about' description (fall back to 'name' if absent)
        course_texts = [
            c.get("about") or c.get("name") or c["course_id"]
            for c in raw_courses
        ]
        c_ids, c_mask = self._tokenize(course_texts, max_length=64)
        print(f"  Courses:   {N_c:,}")

        # ── 3. Videos / resources (real text) ────────────────────────
        # video.json records: {"video_id": str, "name": str, "course_id": str}
        video_ids  = [v["video_id"] for v in raw_videos]
        video_map: Dict[str, int] = {v: i for i, v in enumerate(video_ids)}
        N_r = len(video_ids)

        video_texts = [v.get("name") or v["video_id"] for v in raw_videos]
        r_ids, r_mask = self._tokenize(video_texts, max_length=32)
        print(f"  Videos:    {N_r:,}")

        # ── 4. Behavioral sequences ───────────────────────────────────
        # user-video.json records: {"user_id": str, "video_id": str,
        #                           "watch_time": float, "date": str|int}
        # Aggregate watch_time per (user, week), yielding 1 feature/week.
        # Weeks: integer-divide date by 7 (if date is already a day offset)
        # or extract ISO week if date is a string.
        print("  Building behavioral sequences...")
        week_watch: Dict[int, Dict[int, float]] = {}   # s_idx → {week → total_watch}

        for rec in rel_uv:
            uid  = rec.get("user_id") or rec.get("userId")
            s_i  = student_map.get(uid)
            if s_i is None:
                continue
            raw_date = rec.get("date", 0)
            if isinstance(raw_date, str):
                try:
                    import datetime
                    d = datetime.date.fromisoformat(raw_date[:10])
                    week = d.isocalendar()[1] % self.cfg.behavioral_seq_len
                except Exception:
                    week = 0
            else:
                week = int(raw_date) // 7
            week = min(max(int(week), 0), self.cfg.behavioral_seq_len - 1)
            wt   = float(rec.get("watch_time", 1.0))
            week_watch.setdefault(s_i, {}).setdefault(week, 0.0)
            week_watch[s_i][week] += wt

        # Build (N_s, seq_len, behavioral_input_dim) — 1 raw feature, zero-padded
        x_behav = np.zeros(
            (N_s, self.cfg.behavioral_seq_len, self.cfg.behavioral_input_dim),
            dtype=np.float32)
        for s_i, week_dict in week_watch.items():
            for week, wt in week_dict.items():
                x_behav[s_i, week, 0] = wt

        x_behav[:, :, 0] = np.log1p(x_behav[:, :, 0])
        mx = x_behav[:, :, 0].max()
        if mx > 0:
            x_behav[:, :, 0] /= mx

        # ── 5. Labels ─────────────────────────────────────────────────
        # user-course.json records: {"user_id": str, "course_id": str,
        #                            "completion_rate": float}
        completion: Dict[str, List[float]] = {}
        for rec in rel_uc:
            uid  = rec.get("user_id") or rec.get("userId")
            rate = float(rec.get("completion_rate", 0.0))
            completion.setdefault(uid, []).append(rate)

        # grade: mean completion_rate across all courses, scaled to [0, 4]
        y_grade = np.array([
            np.mean(completion.get(sid, [0.0])) * 4.0
            for sid in student_ids
        ], dtype=np.float32)

        # dropout: users with 0 completed courses (completion_rate == 0 for all)
        y_dropout = np.array([
            1 if all(r == 0.0 for r in completion.get(sid, [0.0])) else 0
            for sid in student_ids
        ], dtype=np.int64)

        # engagement: tertile split of total video watch time
        total_watch = np.array([
            sum(week_watch.get(i, {}).values()) for i in range(N_s)
        ], dtype=np.float32)
        t33, t67 = np.percentile(total_watch, [33, 67])
        y_engagement = np.where(total_watch >= t67, 2,
                        np.where(total_watch >= t33, 1, 0)).astype(np.int64)

        # ── 6. Student text stub (no student text in MOOCCube) ─────────
        s_text_ids        = torch.zeros(N_s, 32, dtype=torch.long)
        s_text_ids[:, 0]  = 101          # [CLS]
        s_text_mask       = torch.zeros(N_s, 32, dtype=torch.long)
        s_text_mask[:, 0] = 1

        # ── 7. Instructor dummy ───────────────────────────────────────
        inst_struct = torch.zeros(1, self.cfg.structured_input_dim)

        # ── 8. Edge: enrolled_in (student → course) ───────────────────
        enroll_src, enroll_dst = [], []
        for rec in rel_uc:
            uid = rec.get("user_id") or rec.get("userId")
            cid = rec.get("course_id") or rec.get("courseId")
            s_i = student_map.get(uid)
            c_i = course_map.get(cid)
            if s_i is not None and c_i is not None:
                enroll_src.append(s_i)
                enroll_dst.append(c_i)

        # Deduplicate
        enroll_pairs = list(set(zip(enroll_src, enroll_dst)))
        if enroll_pairs:
            enroll_src_a, enroll_dst_a = zip(*enroll_pairs)
        else:
            enroll_src_a, enroll_dst_a = [], []
        enroll_ei = torch.from_numpy(
            np.stack([np.array(enroll_src_a), np.array(enroll_dst_a)]).astype(np.int64))

        # ── 9. Edge: collaborated_with (co-enrollment, ≥2 shared) ─────
        if len(enroll_src_a) > 0:
            s_arr = np.array(enroll_src_a)
            c_arr = np.array(enroll_dst_a)
            A_mat = csr_matrix((np.ones(len(s_arr)), (s_arr, c_arr)),
                               shape=(N_s, N_c))
            C_mat = (A_mat @ A_mat.T).tocoo()
            keep  = (C_mat.data >= 2) & (C_mat.row != C_mat.col)
            src_c, dst_c = C_mat.row[keep], C_mat.col[keep]
            if len(src_c) > 50_000:
                perm  = np.random.default_rng(42).choice(len(src_c), 50_000, replace=False)
                src_c, dst_c = src_c[perm], dst_c[perm]
            collab_ei = torch.from_numpy(
                np.stack([src_c, dst_c]).astype(np.int64))
        else:
            collab_ei = torch.zeros(2, 0, dtype=torch.long)

        # ── 10. Edge: accessed (student → video, top 100 per student) ──
        watch_agg: Dict[Tuple[int, int], float] = {}
        for rec in rel_uv:
            uid = rec.get("user_id") or rec.get("userId")
            vid = rec.get("video_id") or rec.get("videoId")
            s_i = student_map.get(uid)
            r_i = video_map.get(vid)
            if s_i is not None and r_i is not None:
                key = (s_i, r_i)
                watch_agg[key] = watch_agg.get(key, 0.0) + float(rec.get("watch_time", 1.0))

        # Sort by watch_time descending and keep top-100 per student
        from collections import defaultdict
        per_student: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for (s_i, r_i), wt in watch_agg.items():
            per_student[s_i].append((wt, r_i))

        acc_src, acc_dst = [], []
        for s_i, pairs in per_student.items():
            for _, r_i in sorted(pairs, reverse=True)[:100]:
                acc_src.append(s_i)
                acc_dst.append(r_i)

        access_ei = torch.from_numpy(
            np.stack([np.array(acc_src), np.array(acc_dst)]).astype(np.int64))

        # ── 11. Edge: submitted_to (student → course via problems) ────
        # user-problem.json records: {"user_id": str, "problem_id": str,
        #                             "course_id": str}  (course_id may be absent)
        # Fall back to mapping problem → course via a best-effort lookup.
        sub_src, sub_dst = [], []
        for rec in rel_up:
            uid = rec.get("user_id") or rec.get("userId")
            cid = rec.get("course_id") or rec.get("courseId")
            s_i = student_map.get(uid)
            c_i = course_map.get(cid) if cid else None
            if s_i is not None and c_i is not None:
                sub_src.append(s_i)
                sub_dst.append(c_i)

        sub_pairs = list(set(zip(sub_src, sub_dst)))
        if sub_pairs:
            sub_src_a, sub_dst_a = zip(*sub_pairs)
            submit_ei = torch.from_numpy(
                np.stack([np.array(sub_src_a),
                          np.array(sub_dst_a)]).astype(np.int64))
        else:
            submit_ei = torch.zeros(2, 0, dtype=torch.long)

        # ── 12. Pack HeteroData ───────────────────────────────────────
        data = HeteroData()

        data["student"].x_struct       = torch.from_numpy(x_struct)
        data["student"].x_behav        = torch.from_numpy(x_behav)
        data["student"].input_ids      = s_text_ids
        data["student"].attention_mask = s_text_mask

        data["course"].input_ids       = c_ids
        data["course"].attention_mask  = c_mask

        data["resource"].input_ids      = r_ids
        data["resource"].attention_mask = r_mask

        data["instructor"].x_struct    = inst_struct

        data["student", "enrolled_in",       "course"].edge_index  = enroll_ei
        data["student", "collaborated_with", "student"].edge_index = collab_ei
        data["student", "accessed",          "resource"].edge_index = access_ei
        data["student", "submitted_to",      "course"].edge_index  = submit_ei

        meta: Dict = {
            "grade":       torch.tensor(y_grade,      dtype=torch.float),
            "dropout":     torch.tensor(y_dropout,    dtype=torch.long),
            "engagement":  torch.tensor(y_engagement, dtype=torch.long),
            "student_ids": np.array(student_ids),
        }

        return data, meta

    # ------------------------------------------------------------------

    def summary(self, data: HeteroData, meta: Dict) -> None:
        W = 60
        print(f"\n{'='*W}")
        print("  MOOCCube Graph Summary")
        print(f"{'='*W}")

        print("\n  Node counts:")
        node_attrs = {"student": "x_struct", "course": "input_ids",
                      "resource": "input_ids", "instructor": "x_struct"}
        for ntype, attr in node_attrs.items():
            n = getattr(data[ntype], attr).shape[0]
            print(f"    {ntype:<15} {n:>8,}")

        print("\n  Edge counts:")
        for (src, rel, dst), ei in data.edge_index_dict.items():
            print(f"    {src} --[{rel}]--> {dst}:  {ei.shape[1]:>8,}")

        print("\n  Label distributions:")
        grade_arr = meta["grade"].numpy()
        print(f"    grade       mean={grade_arr.mean():.3f}  "
              f"std={grade_arr.std():.3f}  "
              f"min={grade_arr.min():.3f}  max={grade_arr.max():.3f}")

        for task in ["dropout", "engagement"]:
            arr = meta[task].numpy()
            vals, cnts = np.unique(arr, return_counts=True)
            parts = "  ".join(
                f"class {int(v)}: {c:,} ({100.*c/len(arr):.1f}%)"
                for v, c in zip(vals, cnts))
            print(f"    {task:<10} {parts}")

        print(f"{'='*W}\n")
