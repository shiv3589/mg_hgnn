# Download from: https://github.com/thukg/MOOCCube
# (NOT huggingface.co/datasets/mooccube — that repo id doesn't exist)
# Direct dataset download (no registration required):
#   http://lfs.aminer.cn/misc/moocdata/data/MOOCCube.zip
# Place the extracted "MOOCCube/" folder's contents at: data/raw/moocc/
#
# ACTUAL on-disk formats (verified by hand against the real release —
# these differ from an idealized/assumed schema in several ways):
#   entities/user.json    — JSON-LINES, one record per line:
#                            {"id": str, "name": str,
#                             "course_order": [course_id, ...],
#                             "enroll_time": [iso_str, ...]}
#   entities/course.json  — JSON-LINES: {"id","name","prerequisites","about"}
#   entities/video.json   — JSON-LINES: {"id","name","start":[...], ...}
#                            (no course_id on the record itself — joined
#                            in via relations/course-video.json)
#   relations/user-video.json    — TAB-separated: user_id\tvideo_id
#                                   (no watch_time/date — just access pairs)
#   relations/user-course.json   — TAB-separated: user_id\tcourse_id
#                                   (no completion_rate — see labels below)
#   relations/course-video.json  — TAB-separated: course_id\tvideo_id
#   relations/user-problem.json  — OPTIONAL. Doesn't exist in the official
#                                   release at all (MOOCCube has no
#                                   problem/exercise entity — that's a
#                                   MOOCCubeX / MOOC-Radar feature, a
#                                   different dataset). When absent,
#                                   submitted_to edges are honestly 0,
#                                   not imputed.
#   additional_information/user_video_act.json — JSON-LINES, the ONLY file
#       with real watch-duration/timing data:
#       {"id": user_id, "activity": [{"course_id","video_id",
#        "watching_count","video_duration","local_watching_time",
#        "video_progress_time","video_start_time","video_end_time",
#        "local_start_time","local_end_time"}, ...]}
#       Covers 48,640 / 199,199 users (24.4%) — the rest have zero logged
#       video activity, which is itself the signal the label proxies
#       below are built on.
#
# LABEL PROXIES — grade/dropout/engagement have no ground truth anywhere
# in MOOCCube (user-course.json is bare enrollment pairs; no
# completion_rate field exists in the official release). Derived instead
# from user_video_act.json, decisions finalized 2026-08-14:
#
#   completion_rate(u, c) = min(1, Σ min(progress_time, video_duration)
#                                    over u's activity in course c
#                                  / Σ video_duration over ALL videos
#                                    belonging to course c)
#   grade(u)      = mean(completion_rate(u, c) for enrolled c) * 4.0
#   dropout(u)    = 1 if mean completion_rate == 0 else 0
#                   (~75.6% positive — this is the genuine floor of this
#                   proxy family: any "< threshold" rule for threshold > 0
#                   is a strict superset of the ==0 group and can only
#                   read HIGHER, never lower. Reported as-is, not tuned
#                   to hit an arbitrary target balance.)
#   engagement(u) = tertile split of total progress_time, cutoffs computed
#                   over users with progress_time > 0 only, then applied
#                   to everyone (zero-activity users fall into class 0)
#
# video_duration is taken from the first-seen value per video_id across
# all activity records. Checked 2026-08-14: only 449/34,101 videos (1.3%)
# have any inconsistency across occurrences, median discrepancy 4.0s —
# benign, first-seen is fine (no need for a median-per-video pass).

import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
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
    "relations/course-video.json": "course→video membership (label denominator)",
    "additional_information/user_video_act.json":
        "per-session watch duration/timing (label + behavioral source)",
}

# Present in the loader's edge-type schema (submitted_to) but absent from
# the real MOOCCube release — degrades gracefully to zero edges rather
# than blocking the whole load.
_OPTIONAL = {
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
        Validate that all required files exist.
        Prints a clear error and download instructions if any are missing.
        Returns True if all required files are present, False otherwise.
        """
        missing = [rel for rel in _REQUIRED if not (self.raw / rel).exists()]
        missing_optional = [rel for rel in _OPTIONAL if not (self.raw / rel).exists()]

        if not missing:
            print("MOOCCube files OK — all required files present.")
            for rel in missing_optional:
                print(f"Note: {rel} not found — submitted_to edges will be empty "
                      f"({_OPTIONAL[rel]} — not present in the official MOOCCube release)")
            return True

        W = 62
        print(f"\n{'!'*W}")
        print("  MOOCCube data files are missing.")
        print(f"  Expected root: {self.raw.resolve()}")
        print(f"\n  Missing files:")
        for rel in missing:
            print(f"    {rel}  ({_REQUIRED[rel]})")
        print(f"\n  Download instructions:")
        print("    Direct download (no registration required):")
        print("      http://lfs.aminer.cn/misc/moocdata/data/MOOCCube.zip")
        print(f"      unzip MOOCCube.zip && cp -r MOOCCube/* {self.raw}/")
        print("    Repo (code/docs only, not the data itself):")
        print("      https://github.com/thukg/MOOCCube")
        print(f"{'!'*W}\n")
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_jsonl(self, rel_path: str):
        """Yields one dict per non-empty line. Every entities/*.json and
        additional_information/*.json file in MOOCCube is JSON-Lines, NOT
        a single JSON array — a plain json.load() raises
        'JSONDecodeError: Extra data' on these files."""
        with open(self.raw / rel_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_tsv(self, rel_path: str) -> List[Tuple[str, str]]:
        """Every relations/*.json file in MOOCCube is actually tab-
        separated plain text (two columns), not JSON."""
        pairs = []
        with open(self.raw / rel_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    def _load_json(self, rel_path: str):
        """Kept for relations/user-problem.json only: that file doesn't
        exist in the official release, so its real format is unverified.
        Assumes a JSON array of records if it's ever present."""
        with open(self.raw / rel_path, encoding="utf-8") as fh:
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

        print("Loading MOOCCube JSON-Lines / TSV files...")
        raw_users   = list(self._load_jsonl("entities/user.json"))
        raw_courses = list(self._load_jsonl("entities/course.json"))
        raw_videos  = list(self._load_jsonl("entities/video.json"))

        rel_uv = self._load_tsv("relations/user-video.json")    # (user_id, video_id)
        rel_uc = self._load_tsv("relations/user-course.json")   # (user_id, course_id)
        rel_cv = self._load_tsv("relations/course-video.json")  # (course_id, video_id)

        prob_path = self.raw / "relations/user-problem.json"
        if prob_path.exists():
            rel_up = self._load_json("relations/user-problem.json")
        else:
            rel_up = []
            print("  user-problem.json absent — submitted_to edges: 0")

        # ── 1. Students ───────────────────────────────────────────────
        student_ids = [u["id"] for u in raw_users]
        student_map: Dict[str, int] = {s: i for i, s in enumerate(student_ids)}
        N_s = len(student_ids)
        print(f"  Students:  {N_s:,}")

        enroll_cnt = np.array(
            [len(u.get("course_order", [])) for u in raw_users], dtype=np.float32)

        video_count_by_user: Dict[str, int] = {}
        for uid, _vid in rel_uv:
            video_count_by_user[uid] = video_count_by_user.get(uid, 0) + 1
        video_cnt = np.array(
            [video_count_by_user.get(sid, 0) for sid in student_ids], dtype=np.float32)

        # No problem/exercise entity exists anywhere in MOOCCube (unlike
        # OULAD) — honestly zero, not fabricated.
        problem_cnt = np.zeros(N_s, dtype=np.float32)

        raw_struct = np.stack([
            np.log1p(enroll_cnt),
            np.log1p(problem_cnt),
            np.log1p(video_cnt),
        ], axis=1)  # (N_s, 3)
        raw_struct = StandardScaler().fit_transform(raw_struct).astype(np.float32)

        x_struct = np.zeros((N_s, self.cfg.structured_input_dim), dtype=np.float32)
        x_struct[:, :raw_struct.shape[1]] = raw_struct

        # ── 2. Courses (real text) ────────────────────────────────────
        course_ids  = [c["id"] for c in raw_courses]
        course_map: Dict[str, int] = {c: i for i, c in enumerate(course_ids)}
        N_c = len(course_ids)

        course_texts = [
            c.get("about") or c.get("name") or c["id"]
            for c in raw_courses
        ]
        c_ids, c_mask = self._tokenize(course_texts, max_length=64)
        print(f"  Courses:   {N_c:,}")

        # ── 3. Videos / resources (real text) ────────────────────────
        video_ids  = [v["id"] for v in raw_videos]
        video_map: Dict[str, int] = {v: i for i, v in enumerate(video_ids)}
        N_r = len(video_ids)

        video_texts = [v.get("name") or v["id"] for v in raw_videos]
        r_ids, r_mask = self._tokenize(video_texts, max_length=32)
        print(f"  Videos:    {N_r:,}")

        # course_id -> [video_id, ...], via relations/course-video.json —
        # needed for the label denominator (course_total_duration) since
        # video.json records carry no course_id of their own.
        course_videos: Dict[str, List[str]] = {}
        for cid, vid in rel_cv:
            course_videos.setdefault(cid, []).append(vid)

        # ── 4. Behavioral sequences + label source ─────────────────────
        # user_video_act.json is the ONLY file with real watch duration/
        # timing — relations/user-video.json is bare (user_id, video_id)
        # pairs with no watch_time or date at all.
        print("  Building behavioral sequences + label proxies from user_video_act.json...")
        video_duration: Dict[str, float] = {}
        week_watch: Dict[int, Dict[int, float]] = {}          # s_idx -> {week -> progress}
        user_course_progress: Dict[Tuple[str, str], float] = {}
        user_total_progress: Dict[str, float] = {}

        act_path = self.raw / "additional_information/user_video_act.json"
        with open(act_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                uid = rec.get("id")
                s_i = student_map.get(uid)

                for act in rec.get("activity", []):
                    vid = act.get("video_id")
                    cid = act.get("course_id")
                    dur = float(act.get("video_duration", 0.0) or 0.0)
                    if vid and dur > 0 and vid not in video_duration:
                        video_duration[vid] = dur  # first-seen; verified benign

                    prog = float(act.get("video_progress_time", 0.0) or 0.0)
                    capped = min(prog, dur) if dur > 0 else 0.0

                    if cid:
                        key = (uid, cid)
                        user_course_progress[key] = user_course_progress.get(key, 0.0) + capped
                    user_total_progress[uid] = user_total_progress.get(uid, 0.0) + prog

                    if s_i is not None:
                        raw_date = act.get("local_start_time") or ""
                        try:
                            import datetime
                            d = datetime.date.fromisoformat(raw_date[:10])
                            week = d.isocalendar()[1] % self.cfg.behavioral_seq_len
                        except Exception:
                            week = 0
                        week = min(max(int(week), 0), self.cfg.behavioral_seq_len - 1)
                        week_watch.setdefault(s_i, {}).setdefault(week, 0.0)
                        week_watch[s_i][week] += prog

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

        # course_total_duration -- denominator for completion_rate, summed
        # over every video belonging to the course (not just watched ones)
        course_total_duration: Dict[str, float] = {
            cid: sum(video_duration.get(v, 0.0) for v in vids)
            for cid, vids in course_videos.items()
        }

        def _completion_rate(uid: str, cid: str) -> float:
            denom = course_total_duration.get(cid, 0.0)
            if denom <= 0:
                return 0.0
            return min(user_course_progress.get((uid, cid), 0.0) / denom, 1.0)

        # ── 5. Labels (proxies — see module docstring) ─────────────────
        user_enrolled: Dict[str, List[str]] = {}
        for uid, cid in rel_uc:
            user_enrolled.setdefault(uid, []).append(cid)

        y_grade = np.zeros(N_s, dtype=np.float32)
        y_dropout = np.zeros(N_s, dtype=np.int64)
        for i, sid in enumerate(student_ids):
            cids = user_enrolled.get(sid, [])
            if cids:
                rates = [_completion_rate(sid, c) for c in cids]
                mean_rate = float(np.mean(rates))
            else:
                mean_rate = 0.0
            y_grade[i] = mean_rate * 4.0
            y_dropout[i] = 1 if mean_rate == 0.0 else 0

        total_watch = np.array(
            [user_total_progress.get(sid, 0.0) for sid in student_ids], dtype=np.float32)
        nonzero = total_watch[total_watch > 0]
        if len(nonzero) > 0:
            t33, t67 = np.percentile(nonzero, [33, 67])
        else:
            t33, t67 = 0.0, 0.0
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
        for uid, cid in rel_uc:
            s_i = student_map.get(uid)
            c_i = course_map.get(cid)
            if s_i is not None and c_i is not None:
                enroll_src.append(s_i)
                enroll_dst.append(c_i)

        enroll_pairs = list(set(zip(enroll_src, enroll_dst)))
        if enroll_pairs:
            enroll_src_a, enroll_dst_a = zip(*enroll_pairs)
        else:
            enroll_src_a, enroll_dst_a = [], []
        enroll_ei = torch.from_numpy(
            np.stack([np.array(enroll_src_a), np.array(enroll_dst_a)]).astype(np.int64))

        # ── 9. Edge: collaborated_with (co-enrollment, per-course capped) ─
        # MOOCCube's real course sizes are large enough (up to 32,914
        # enrolled students in one course) that the original A_mat @
        # A_mat.T sparse matmul (inherited from OULAD, where courses are
        # tiny) tries to materialize ~5 BILLION intermediate nonzeros
        # (37.3GB) before any filtering applies -- confirmed OOM on this
        # 30GB instance 2026-08-14. Fixed by subsampling any course above
        # CAP=300 enrolled students down to 300 BEFORE generating pairs.
        # This also drops the original's ">=2 shared courses" filter and
        # its 50,000-edge final sample, in favor of a single 5,000,000-
        # edge hard safety cap applied during generation (approved
        # 2026-08-14).
        from collections import defaultdict

        course_to_students: Dict[int, List[int]] = defaultdict(list)
        for s_i, c_i in zip(enroll_src_a, enroll_dst_a):
            course_to_students[c_i].append(s_i)

        COLLAB_CAP = 300
        rng = np.random.default_rng(seed=42)  # reproducible
        capped_course_to_students: Dict[int, List[int]] = {}
        n_subsampled = 0
        for c_i, students in course_to_students.items():
            if len(students) > COLLAB_CAP:
                students = rng.choice(students, size=COLLAB_CAP, replace=False).tolist()
                n_subsampled += 1
            capped_course_to_students[c_i] = students

        print(f"  collaborated_with: {n_subsampled}/{len(course_to_students)} "
              f"courses subsampled to cap={COLLAB_CAP}")

        collab_pairs: List[Tuple[int, int]] = []
        HARD_CAP = 5_000_000
        for c_i, students in capped_course_to_students.items():
            for i in range(len(students)):
                for j in range(i + 1, len(students)):
                    collab_pairs.append((students[i], students[j]))
            if len(collab_pairs) > HARD_CAP:
                print(f"  collaborated_with: hard cap hit at {len(collab_pairs):,} pairs")
                break

        if collab_pairs:
            src_c = np.array([p[0] for p in collab_pairs])
            dst_c = np.array([p[1] for p in collab_pairs])
            collab_ei = torch.from_numpy(np.stack([src_c, dst_c]).astype(np.int64))
        else:
            collab_ei = torch.zeros(2, 0, dtype=torch.long)

        # ── 10. Edge: accessed (student → video) ────────────────────────
        # relations/user-video.json has no watch_time (Fix 4), so unlike
        # OULAD there's no ranking signal to keep only a top-100 subset —
        # every distinct (student, video) pair is included.
        acc_src, acc_dst = [], []
        seen_pairs = set()
        for uid, vid in rel_uv:
            s_i = student_map.get(uid)
            r_i = video_map.get(vid)
            if s_i is None or r_i is None:
                continue
            key = (s_i, r_i)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            acc_src.append(s_i)
            acc_dst.append(r_i)

        access_ei = torch.from_numpy(
            np.stack([np.array(acc_src), np.array(acc_dst)]).astype(np.int64)) \
            if acc_src else torch.zeros(2, 0, dtype=torch.long)

        # ── 11. Edge: submitted_to (student → course via problems) ────
        # rel_up is [] unless relations/user-problem.json exists (it
        # doesn't, in the official release) — honestly empty, not
        # fabricated. See module docstring / Fix 7.
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
