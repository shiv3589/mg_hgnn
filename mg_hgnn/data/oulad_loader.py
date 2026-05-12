import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer

from config import Config


class OULADLoader:

    def __init__(self, config: Config, raw_dir: str = "data/raw/oulad"):
        self.cfg = config
        self.raw = pathlib.Path(raw_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        Returns (data, meta) where meta holds label tensors and the
        original student_id array for downstream evaluation.
        """
        print("Loading OULAD tables...")
        info      = pd.read_csv(self.raw / "studentInfo.csv")
        reg       = pd.read_csv(self.raw / "studentRegistration.csv")
        vle_log   = pd.read_csv(self.raw / "studentVle.csv")
        assess    = pd.read_csv(self.raw / "studentAssessment.csv")
        courses   = pd.read_csv(self.raw / "courses.csv")
        vle_meta  = pd.read_csv(self.raw / "vle.csv")
        asmt_meta = pd.read_csv(self.raw / "assessments.csv")

        # ── Student index ─────────────────────────────────────────────
        # One node per unique id_student; keep the most recent presentation.
        info_dedup = (info.sort_values("code_presentation", ascending=False)
                         .drop_duplicates(subset="id_student")
                         .reset_index(drop=True))
        student_ids = info_dedup["id_student"].values
        student_map: Dict[int, int] = {int(s): i for i, s in enumerate(student_ids)}
        N_s = len(student_ids)
        print(f"  Students:  {N_s:,}")

        # ── Structured features ───────────────────────────────────────
        cat_cols = ["gender", "region", "highest_education",
                    "imd_band", "age_band", "disability",
                    "code_module", "code_presentation"]
        for col in cat_cols:
            info_dedup[col] = info_dedup[col].fillna("Unknown")
        cat_dummies = (pd.get_dummies(info_dedup[cat_cols], drop_first=False)
                         .values.astype(np.float32))

        # Aggregate VLE statistics per student
        vle_agg = (vle_log.groupby("id_student")
                          .agg(total_clicks=("sum_click", "sum"),
                               active_days=("date", "nunique"),
                               distinct_sites=("id_site", "nunique"))
                          .reset_index())

        # Aggregate assessment statistics per student
        assess_agg = (assess.groupby("id_student")
                            .agg(mean_score=("score", "mean"),
                                 num_submitted=("id_assessment", "count"))
                            .reset_index())

        # Registration statistics per student
        reg_first = (reg.drop_duplicates("id_student")
                        [["id_student", "date_registration", "date_unregistration"]]
                        .copy())
        reg_first["is_unregistered"] = reg_first["date_unregistration"].notna().astype(float)

        feat_df = info_dedup[["id_student",
                               "num_of_prev_attempts", "studied_credits"]].copy()
        feat_df = feat_df.merge(vle_agg,    on="id_student", how="left")
        feat_df = feat_df.merge(assess_agg, on="id_student", how="left")
        feat_df = feat_df.merge(
            reg_first[["id_student", "date_registration", "is_unregistered"]],
            on="id_student", how="left")
        feat_df = feat_df.fillna(0)
        feat_df["total_clicks"] = np.log1p(feat_df["total_clicks"])

        num_cols = ["num_of_prev_attempts", "studied_credits",
                    "total_clicks", "active_days", "distinct_sites",
                    "mean_score", "num_submitted",
                    "date_registration", "is_unregistered"]
        num_arr  = feat_df[num_cols].values.astype(np.float32)

        raw_feat = np.concatenate([num_arr, cat_dummies], axis=1)
        raw_feat = StandardScaler().fit_transform(raw_feat).astype(np.float32)

        n_raw    = raw_feat.shape[1]
        x_struct = np.zeros((N_s, self.cfg.structured_input_dim), dtype=np.float32)
        x_struct[:, :min(n_raw, self.cfg.structured_input_dim)] = \
            raw_feat[:, :self.cfg.structured_input_dim]
        print(f"  Structured: {n_raw} raw features → padded to {self.cfg.structured_input_dim}")

        # ── Student labels ────────────────────────────────────────────
        grade_map   = {"Distinction": 4.0, "Pass": 3.0, "Fail": 1.0, "Withdrawn": 0.0}
        dropout_map = {"Withdrawn": 1, "Fail": 0, "Pass": 0, "Distinction": 0}
        engage_map  = {"Distinction": 2, "Pass": 1, "Fail": 0, "Withdrawn": 0}

        fr = info_dedup["final_result"].fillna("Fail")
        y_grade      = np.array([grade_map.get(r, 1.0) for r in fr],  dtype=np.float32)
        y_dropout    = np.array([dropout_map.get(r, 0)  for r in fr], dtype=np.int64)
        y_engagement = np.array([engage_map.get(r, 0)   for r in fr], dtype=np.int64)

        # ── Student text stub (no text in OULAD → [CLS]-only) ─────────
        text_len    = 32
        s_text_ids  = torch.zeros(N_s, text_len, dtype=torch.long)
        s_text_ids[:, 0] = 101          # [CLS] token id in BERT vocab
        s_text_mask = torch.zeros(N_s, text_len, dtype=torch.long)
        s_text_mask[:, 0] = 1           # attend only to [CLS]

        # ── Behavioral sequences ──────────────────────────────────────
        print("  Building behavioral sequences (~30 s for 10 M rows)...")
        vle_typed = vle_log.merge(
            vle_meta[["id_site", "activity_type"]], on="id_site", how="left")
        vle_typed["activity_type"] = vle_typed["activity_type"].fillna("unknown")

        # Choose top-K activity types as feature channels
        top_acts   = (vle_typed.groupby("activity_type")["sum_click"]
                               .sum()
                               .sort_values(ascending=False)
                               .index[:self.cfg.behavioral_input_dim]
                               .tolist())
        act_to_idx = {a: i for i, a in enumerate(top_acts)}

        vle_typed["week"]  = (vle_typed["date"] // 7).clip(
            0, self.cfg.behavioral_seq_len - 1).astype(int)
        vle_typed["s_idx"] = vle_typed["id_student"].map(student_map)
        vle_typed["a_idx"] = vle_typed["activity_type"].map(act_to_idx)

        valid = (vle_typed.dropna(subset=["s_idx", "a_idx"])
                          .query(f"s_idx < {N_s} and "
                                 f"a_idx < {self.cfg.behavioral_input_dim}"))

        agg_beh = (valid.groupby(["s_idx", "week", "a_idx"])["sum_click"]
                        .sum().reset_index())

        x_behav = np.zeros(
            (N_s, self.cfg.behavioral_seq_len, self.cfg.behavioral_input_dim),
            dtype=np.float32)
        x_behav[agg_beh["s_idx"].astype(int).values,
                agg_beh["week"].astype(int).values,
                agg_beh["a_idx"].astype(int).values] = (
            agg_beh["sum_click"].values.astype(np.float32))  # no dupes after groupby

        x_behav = np.log1p(x_behav)
        ch_max  = x_behav.max(axis=(0, 1), keepdims=True)
        x_behav /= np.where(ch_max == 0, 1.0, ch_max)
        print(f"  Behavioral tensor: {x_behav.shape}")

        # ── Course nodes ──────────────────────────────────────────────
        course_keys = (courses.drop_duplicates(["code_module", "code_presentation"])
                               .reset_index(drop=True))
        course_map  = {(r.code_module, r.code_presentation): i
                       for i, r in course_keys.iterrows()}
        N_c = len(course_keys)
        course_texts = [f"{r.code_module} {r.code_presentation}"
                        for _, r in course_keys.iterrows()]
        c_ids, c_mask = self._tokenize(course_texts, max_length=text_len)
        print(f"  Courses:   {N_c}")

        # ── Resource nodes (VLE sites) ────────────────────────────────
        res_keys   = (vle_meta[["id_site", "activity_type"]]
                               .drop_duplicates("id_site")
                               .reset_index(drop=True))
        resource_map = {int(r.id_site): i for i, r in res_keys.iterrows()}
        N_r = len(res_keys)
        res_texts = res_keys["activity_type"].fillna("unknown").tolist()
        r_ids, r_mask = self._tokenize(res_texts, max_length=text_len)
        print(f"  Resources: {N_r:,}")

        # ── Instructor (dummy) ────────────────────────────────────────
        inst_struct = torch.zeros(1, self.cfg.structured_input_dim)

        # ── Edge: enrolled_in (student → course) ─────────────────────
        reg_ei = reg[reg["id_student"].isin(student_map)].copy()
        reg_ei["s_idx"] = reg_ei["id_student"].map(student_map)
        reg_ei["c_idx"] = reg_ei.apply(
            lambda r: course_map.get((r.code_module, r.code_presentation), -1), axis=1)
        enroll = reg_ei[reg_ei["c_idx"] >= 0][["s_idx", "c_idx"]].drop_duplicates()
        enroll_ei = torch.from_numpy(
            np.stack([enroll["s_idx"].values, enroll["c_idx"].values]).astype(np.int64))

        # ── Edge: collaborated_with (student ↔ student via co-enrollment) ──
        s_arr = enroll["s_idx"].values
        c_arr = enroll["c_idx"].values
        A_mat = csr_matrix((np.ones(len(s_arr), dtype=np.float32),
                            (s_arr, c_arr)), shape=(N_s, N_c))
        C_mat = (A_mat @ A_mat.T).tocoo()
        keep  = (C_mat.data >= 2) & (C_mat.row != C_mat.col)
        src_c, dst_c = C_mat.row[keep], C_mat.col[keep]
        if len(src_c) > 50_000:
            perm  = np.random.default_rng(42).choice(len(src_c), 50_000, replace=False)
            src_c, dst_c = src_c[perm], dst_c[perm]
        collab_ei = torch.from_numpy(
            np.stack([src_c, dst_c]).astype(np.int64))

        # ── Edge: accessed (student → resource, top-50 per student) ──
        vle_s = vle_log[vle_log["id_student"].isin(student_map)].copy()
        vle_s["s_idx"] = vle_s["id_student"].map(student_map)
        vle_s["r_idx"] = vle_s["id_site"].map(resource_map)
        vle_s = vle_s.dropna(subset=["s_idx", "r_idx"])
        access_top = (vle_s.groupby(["s_idx", "r_idx"])["sum_click"]
                           .sum().reset_index()
                           .sort_values("sum_click", ascending=False)
                           .groupby("s_idx").head(50))
        access_ei = torch.from_numpy(
            np.stack([access_top["s_idx"].values,
                      access_top["r_idx"].values]).astype(np.int64))

        # ── Edge: submitted_to (student → course via assessments) ─────
        assess_full = assess.merge(
            asmt_meta[["id_assessment", "code_module", "code_presentation"]],
            on="id_assessment", how="left")
        assess_full = assess_full[assess_full["id_student"].isin(student_map)].copy()
        assess_full["s_idx"] = assess_full["id_student"].map(student_map)
        assess_full["c_idx"] = assess_full.apply(
            lambda r: course_map.get((r.code_module, r.code_presentation), -1), axis=1)
        submit = (assess_full[assess_full["c_idx"] >= 0]
                  [["s_idx", "c_idx"]].drop_duplicates())
        submit_ei = torch.from_numpy(
            np.stack([submit["s_idx"].values,
                      submit["c_idx"].values]).astype(np.int64))

        # ── Pack HeteroData ───────────────────────────────────────────
        data = HeteroData()

        data["student"].num_nodes      = N_s
        data["student"].x_struct       = torch.from_numpy(x_struct)
        data["student"].x_behav        = torch.from_numpy(x_behav)
        data["student"].input_ids      = s_text_ids
        data["student"].attention_mask = s_text_mask

        data["course"].num_nodes       = N_c
        data["course"].input_ids       = c_ids
        data["course"].attention_mask  = c_mask

        data["resource"].num_nodes     = N_r
        data["resource"].input_ids      = r_ids
        data["resource"].attention_mask = r_mask

        data["instructor"].num_nodes   = 1
        data["instructor"].x_struct    = inst_struct

        data["student", "enrolled_in",       "course"].edge_index  = enroll_ei
        data["student", "collaborated_with", "student"].edge_index = collab_ei
        data["student", "accessed",          "resource"].edge_index = access_ei
        data["student", "submitted_to",      "course"].edge_index  = submit_ei

        meta: Dict = {
            "grade":       torch.tensor(y_grade,      dtype=torch.float),
            "dropout":     torch.tensor(y_dropout,    dtype=torch.long),
            "engagement":  torch.tensor(y_engagement, dtype=torch.long),
            "student_ids": student_ids,
        }

        return data, meta

    # ------------------------------------------------------------------

    def summary(self, data: HeteroData, meta: Dict) -> None:
        W = 60
        print(f"\n{'='*W}")
        print("  OULAD Graph Summary")
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
        for val, label in [(4., "Distinction"), (3., "Pass"),
                           (1., "Fail"),        (0., "Withdrawn")]:
            cnt = int((grade_arr == val).sum())
            print(f"    grade      {label:<14} {cnt:>7,}  ({100.*cnt/len(grade_arr):.1f}%)")

        for task in ["dropout", "engagement"]:
            arr  = meta[task].numpy()
            vals, cnts = np.unique(arr, return_counts=True)
            parts = "  ".join(
                f"class {int(v)}: {c:,} ({100.*c/len(arr):.1f}%)"
                for v, c in zip(vals, cnts))
            print(f"    {task:<10} {parts}")

        print(f"{'='*W}\n")
