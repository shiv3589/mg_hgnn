"""
ASSISTmentsLoader — builds HeteroData for ASSISTments 2009 and 2015.

Graph schema mirrors OULADLoader so MG_HGNN and all baselines work unchanged:
  Node types : student, course (=skill), resource (=problem), instructor (dummy)
  Edge types : enrolled_in       student → course   (student practiced this skill)
               collaborated_with student → student  (share ≥1 skill, cap 30K)
               accessed          student → resource (top-50 problems per student)
               submitted_to      student → course   (any submission to this skill)

ASSISTments has no text — all text_ids are zero-padded with a [CLS] token.
Behavioral sequences use row-order as a temporal proxy, binned into
cfg.behavioral_seq_len windows; channel 0=correct, channel 1=hint_used.

Download instructions (printed by check_files()):
  2009: https://sites.google.com/site/assistmentsdata/home/
        2009-2010-assistment-data/skill-builder-data-2009-2010
        save as: data/raw/assistments09/skill_builder_data_corrected.csv

  2015: https://sites.google.com/site/assistmentsdata/datasets/
        2015-assistments-skill-builder-data
        save as: data/raw/assistments15/2015_skill_builder_data_corrected.csv
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import HeteroData

from config import Config


class ASSISTmentsLoader:

    def __init__(self, config: Config, version: str = '2015'):
        self.cfg = config
        self.version = version
        if version == '2009':
            self.raw = Path(
                'data/raw/assistments09/skill_builder_data_corrected.csv')
        else:
            self.raw = Path(
                'data/raw/assistments15/2015_skill_builder_data_corrected.csv')

    def check_files(self) -> bool:
        if not self.raw.exists():
            print(f'File not found: {self.raw}')
            print('Download from:')
            if self.version == '2009':
                print('  https://sites.google.com/site/assistmentsdata/home/'
                      '2009-2010-assistment-data/skill-builder-data-2009-2010')
            else:
                print('  https://sites.google.com/site/assistmentsdata/datasets/'
                      '2015-assistments-skill-builder-data')
            return False
        print(f'ASSISTments {self.version}: found ✓')
        return True

    def load(self) -> Tuple[HeteroData, Dict]:
        print(f'Loading ASSISTments {self.version}...')
        df = pd.read_csv(self.raw, encoding='latin1', low_memory=False)

        # ── 1. Normalise schema ───────────────────────────────────────
        # 2015 dataset has only 4 cols: user_id, log_id, sequence_id, correct.
        # Map sequence_id → problem_id and skill_name so the rest of the
        # pipeline is version-agnostic.
        if 'problem_id' not in df.columns:
            # 2015: use sequence_id (100 skill builders) as problem proxy
            df['problem_id'] = df['sequence_id']
        if 'skill_name' not in df.columns:
            df['skill_name'] = df['sequence_id'].astype(str)

        # ── 1b. Clean ─────────────────────────────────────────────────
        df = df.dropna(subset=['user_id', 'correct', 'problem_id'])
        df['correct']    = pd.to_numeric(df['correct'],    errors='coerce')
        df['user_id']    = pd.to_numeric(df['user_id'],    errors='coerce')
        df['problem_id'] = pd.to_numeric(df['problem_id'], errors='coerce')
        df = df.dropna(subset=['correct', 'user_id', 'problem_id'])
        df['correct']    = df['correct'].astype(int)
        df['user_id']    = df['user_id'].astype(int)
        df['problem_id'] = df['problem_id'].astype(int)

        if 'attempt_count' in df.columns:
            df['attempt_count'] = (pd.to_numeric(df['attempt_count'], errors='coerce')
                                   .fillna(1).astype(int))
        else:
            df['attempt_count'] = 1

        if 'hint_count' in df.columns:
            df['hint_count'] = (pd.to_numeric(df['hint_count'], errors='coerce')
                                .fillna(0).astype(int))
        else:
            df['hint_count'] = 0

        df['skill_name'] = df['skill_name'].fillna('unknown')

        print(f'  Raw rows after cleaning: {len(df):,}')

        # ── 2. Student index ──────────────────────────────────────────
        student_ids = np.array(sorted(df['user_id'].unique()))
        student_map = {int(s): i for i, s in enumerate(student_ids)}
        N_s = len(student_ids)
        df['s_idx'] = df['user_id'].map(student_map)
        print(f'  Students:  {N_s:,}')

        # ── 3. Skill index (course) ───────────────────────────────────
        skill_names = sorted(df['skill_name'].unique())
        skill_map   = {s: i for i, s in enumerate(skill_names)}
        N_sk = len(skill_names)
        df['sk_idx'] = df['skill_name'].map(skill_map)
        print(f'  Skills (courses): {N_sk:,}')

        # ── 4. Problem index (resource) ───────────────────────────────
        problem_ids = np.array(sorted(df['problem_id'].unique()))
        problem_map = {int(p): i for i, p in enumerate(problem_ids)}
        N_p = len(problem_ids)
        df['p_idx'] = df['problem_id'].map(problem_map)
        print(f'  Problems (resources): {N_p:,}')

        # ── 5. Student structured features ────────────────────────────
        # total_correct, total_attempts, avg_hint, n_problems, skill_diversity
        stu_agg = df.groupby('s_idx').agg(
            total_correct =('correct',      'sum'),
            total_attempts=('attempt_count','sum'),
            avg_hint      =('hint_count',   'mean'),
            n_problems    =('problem_id',   'nunique'),
        ).reindex(range(N_s), fill_value=0).reset_index(drop=True)

        skill_div = (df.groupby('s_idx')['skill_name']
                       .nunique()
                       .reindex(range(N_s), fill_value=0))
        stu_agg['skill_diversity'] = skill_div.values

        feat_arr = stu_agg[['total_correct', 'total_attempts', 'avg_hint',
                             'n_problems', 'skill_diversity']].values.astype(np.float32)
        feat_arr = StandardScaler().fit_transform(feat_arr)

        x_struct = np.zeros((N_s, self.cfg.structured_input_dim), dtype=np.float32)
        x_struct[:, :5] = feat_arr
        print(f'  Structured: 5 features → padded to {self.cfg.structured_input_dim}')

        # ── 6. Student labels ─────────────────────────────────────────
        # grade: average correct rate per student (0–1)
        y_grade = (df.groupby('s_idx')['correct']
                     .mean()
                     .reindex(range(N_s), fill_value=0.0)
                     .values.astype(np.float32))

        # dropout: students with fewer than 10 total attempts
        total_att = (df.groupby('s_idx')['attempt_count']
                       .sum()
                       .reindex(range(N_s), fill_value=0)
                       .values)
        y_dropout = (total_att < 10).astype(np.int64)

        # engagement: tertile of distinct problems seen (0=low,1=mid,2=high)
        n_probs = (df.groupby('s_idx')['problem_id']
                     .nunique()
                     .reindex(range(N_s), fill_value=0)
                     .values)
        t33, t67 = np.percentile(n_probs, [33, 67])
        y_engagement = np.clip(np.digitize(n_probs, [t33, t67]), 0, 2).astype(np.int64)

        # ── 7. Student text stubs (no text in ASSISTments) ────────────
        text_len    = 32
        s_text_ids  = torch.zeros(N_s, text_len, dtype=torch.long)
        s_text_ids[:, 0] = 101          # [CLS]
        s_text_mask = torch.zeros(N_s, text_len, dtype=torch.long)
        s_text_mask[:, 0] = 1

        # ── 8. Student behavioral sequences ──────────────────────────
        # Use row order as temporal proxy; bin into behavioral_seq_len windows.
        # Channel 0 = correct, channel 1 = hint_used.
        print('  Building behavioral sequences...')
        df = df.reset_index(drop=True)
        df['row_rank'] = df.groupby('s_idx').cumcount()
        max_rank       = df.groupby('s_idx')['row_rank'].transform('max').clip(lower=0)
        df['time_bin'] = (
            (df['row_rank'] / (max_rank + 1)) * self.cfg.behavioral_seq_len
        ).clip(0, self.cfg.behavioral_seq_len - 1).astype(int)

        beh_agg = (df.groupby(['s_idx', 'time_bin'])
                     .agg(avg_correct=('correct',    'mean'),
                          avg_hint   =('hint_count', 'mean'))
                     .reset_index())

        x_behav = np.zeros(
            (N_s, self.cfg.behavioral_seq_len, self.cfg.behavioral_input_dim),
            dtype=np.float32)
        s_arr = beh_agg['s_idx'].values.astype(int)
        t_arr = beh_agg['time_bin'].values.astype(int)
        x_behav[s_arr, t_arr, 0] = beh_agg['avg_correct'].values.astype(np.float32)
        x_behav[s_arr, t_arr, 1] = beh_agg['avg_hint'].values.astype(np.float32)
        print(f'  Behavioral tensor: {x_behav.shape}')

        # ── 9. Text stubs for course/resource nodes ───────────────────
        c_text_ids  = torch.zeros(N_sk, text_len, dtype=torch.long)
        c_text_ids[:, 0] = 101
        c_text_mask = torch.zeros(N_sk, text_len, dtype=torch.long)
        c_text_mask[:, 0] = 1

        r_text_ids  = torch.zeros(N_p, text_len, dtype=torch.long)
        r_text_ids[:, 0] = 101
        r_text_mask = torch.zeros(N_p, text_len, dtype=torch.long)
        r_text_mask[:, 0] = 1

        # ── 10. Edges ─────────────────────────────────────────────────

        # enrolled_in: student → skill (unique pairs)
        enroll = df[['s_idx', 'sk_idx']].drop_duplicates()
        enroll_ei = torch.from_numpy(
            np.stack([enroll['s_idx'].values,
                      enroll['sk_idx'].values]).astype(np.int64))

        # submitted_to: student → skill (same as enrolled_in but explicit semantics)
        submit = df[['s_idx', 'sk_idx']].drop_duplicates()
        submit_ei = torch.from_numpy(
            np.stack([submit['s_idx'].values,
                      submit['sk_idx'].values]).astype(np.int64))

        # accessed: student → problem, top-50 per student by attempt count
        sp_counts = (df.groupby(['s_idx', 'p_idx'])['attempt_count']
                       .sum().reset_index()
                       .sort_values('attempt_count', ascending=False)
                       .groupby('s_idx').head(50))
        access_ei = torch.from_numpy(
            np.stack([sp_counts['s_idx'].values,
                      sp_counts['p_idx'].values]).astype(np.int64))

        # collaborated_with: student-student sharing ≥1 skill, capped at 30K
        A_mat = csr_matrix(
            (np.ones(len(enroll), dtype=np.float32),
             (enroll['s_idx'].values, enroll['sk_idx'].values)),
            shape=(N_s, N_sk))
        C_mat = (A_mat @ A_mat.T).tocoo()
        keep  = (C_mat.data >= 2) & (C_mat.row != C_mat.col)
        src_c, dst_c = C_mat.row[keep], C_mat.col[keep]
        if len(src_c) > 30_000:
            perm  = np.random.default_rng(42).choice(len(src_c), 30_000, replace=False)
            src_c, dst_c = src_c[perm], dst_c[perm]
        collab_ei = torch.from_numpy(
            np.stack([src_c, dst_c]).astype(np.int64))

        print(f'  enrolled_in:       {enroll_ei.shape[1]:,} edges')
        print(f'  submitted_to:      {submit_ei.shape[1]:,} edges')
        print(f'  accessed:          {access_ei.shape[1]:,} edges')
        print(f'  collaborated_with: {collab_ei.shape[1]:,} edges')

        # ── 11. Pack HeteroData ───────────────────────────────────────
        data = HeteroData()

        data['student'].num_nodes      = N_s
        data['student'].x_struct       = torch.from_numpy(x_struct)
        data['student'].x_behav        = torch.from_numpy(x_behav)
        data['student'].input_ids      = s_text_ids
        data['student'].attention_mask = s_text_mask

        data['course'].num_nodes       = N_sk   # skills mapped to 'course'
        data['course'].input_ids       = c_text_ids
        data['course'].attention_mask  = c_text_mask

        data['resource'].num_nodes     = N_p    # problems mapped to 'resource'
        data['resource'].input_ids     = r_text_ids
        data['resource'].attention_mask = r_text_mask

        data['instructor'].num_nodes   = 1
        data['instructor'].x_struct    = torch.zeros(1, self.cfg.structured_input_dim)

        data['student', 'enrolled_in',       'course'].edge_index  = enroll_ei
        data['student', 'collaborated_with', 'student'].edge_index = collab_ei
        data['student', 'accessed',          'resource'].edge_index = access_ei
        data['student', 'submitted_to',      'course'].edge_index  = submit_ei

        meta: Dict = {
            'grade':       torch.tensor(y_grade,      dtype=torch.float),
            'dropout':     torch.tensor(y_dropout,    dtype=torch.long),
            'engagement':  torch.tensor(y_engagement, dtype=torch.long),
            'student_ids': student_ids,
        }

        return data, meta

    def summary(self, data: HeteroData, meta: Dict) -> None:
        W = 60
        print(f"\n{'='*W}")
        print(f'  ASSISTments {self.version} Graph Summary')
        print(f"{'='*W}")

        print('\n  Node counts:')
        for ntype in data.node_types:
            print(f'    {ntype:<15} {data[ntype].num_nodes:>8,}')

        print('\n  Edge counts:')
        for (src, rel, dst), ei in data.edge_index_dict.items():
            print(f'    {src} --[{rel}]--> {dst}:  {ei.shape[1]:>8,}')

        print('\n  Label distributions:')
        drop_arr = meta['dropout'].numpy()
        print(f'    dropout    pos={drop_arr.sum():,}  ({100.*drop_arr.mean():.1f}%)')
        grade_arr = meta['grade'].numpy()
        print(f'    grade      mean={grade_arr.mean():.3f}  std={grade_arr.std():.3f}')
        eng_arr = meta['engagement'].numpy()
        for v in [0, 1, 2]:
            cnt = int((eng_arr == v).sum())
            print(f'    engagement class {v}: {cnt:,} ({100.*cnt/len(eng_arr):.1f}%)')

        print(f"{'='*W}\n")
