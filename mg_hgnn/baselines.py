"""
Baseline models for MG-HGNN comparison.

All four baselines share the same interface:
    fit(data: HeteroData, train_idx: np.ndarray) -> None
    predict(data: HeteroData, test_idx: np.ndarray) -> dict
        returns {'grade': np.ndarray, 'dropout': np.ndarray, 'engagement': np.ndarray}

grade    → raw float predictions (for RMSE)
dropout  → probabilities in [0,1]  (for AUC-ROC)
engagement → integer class labels  (for Engagement-F1)
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HGTConv, HANConv
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from config import Config
from models.encoders import StructuredEncoder, TextEncoder, BehavioralEncoder


# ----------------------------------------------------------------
# Constants shared by all neural baselines
# ----------------------------------------------------------------

_TRAIN_EPOCHS       = 30    # full end-to-end epochs (GAT)
_HGT_WARMUP_EPOCHS  = 3     # full-graph warmup before head-only training
_HAN_WARMUP_EPOCHS  = 3
_TRAIN_HEAD_EPOCHS  = 200   # head-only epoch budget (HGT, HAN) — matches
                             # MG-HGNN's cfg.epochs; dual-patience stopping
                             # below almost always ends the run earlier
_HEAD_PATIENCE      = 20    # dual (AUC + F1) patience — matches cfg.patience
                             # and train_fast.py's early-stop rule exactly
_HEAD_VAL_FRAC      = 0.2   # inner val split carved from train_idx
_HAN_MAX_ACCESSED   = 10_000  # cap accessed edges before building SRS meta-path
_BERT_CACHE_PATH    = "data/bert_cache_oulad.pt"

_LR            = 1e-3
_WEIGHT_DECAY  = 1e-4
_LAMBDA_GRADE  = 0.4
_LAMBDA_DROP   = 0.4
_LAMBDA_ENG    = 0.2

_EDGE_TRIPLES = [
    ("student", "enrolled_in",       "course"),
    ("student", "collaborated_with", "student"),
    ("student", "accessed",          "resource"),
    ("student", "submitted_to",      "course"),
]


# ----------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------

def _early_student_features(raw_dir: str = "data/raw/oulad") -> np.ndarray:
    """
    Early-semester (first 4 weeks) student features for XGBoost — no leakage.

    Replicates OULADLoader's student deduplication so row i here maps to
    node i in the HeteroData.  Features include only information observable
    BEFORE week 4 of the course:
      - Demographics: gender, imd_band, age_band, disability, highest_education
      - num_of_prev_attempts, studied_credits (registered amount)
      - log(1 + sum_click) for VLE interactions with date < 28
    Full-semester aggregates (total_clicks, active_days, mean_score, etc.)
    and the is_unregistered flag are intentionally excluded.
    """
    raw = pathlib.Path(raw_dir)
    info    = pd.read_csv(raw / "studentInfo.csv")
    vle_log = pd.read_csv(raw / "studentVle.csv")

    # Mirror OULADLoader dedup: keep most-recent presentation per student
    info_dedup = (info.sort_values("code_presentation", ascending=False)
                      .drop_duplicates(subset="id_student")
                      .reset_index(drop=True))

    # Safe demographics — available at enrollment, no semester data
    cat_cols = ["gender", "imd_band", "age_band", "disability", "highest_education"]
    for col in cat_cols:
        info_dedup[col] = info_dedup[col].fillna("Unknown")
    cat_dummies = (pd.get_dummies(info_dedup[cat_cols], drop_first=False)
                     .values.astype(np.float32))

    # Early VLE activity: first 4 weeks only (date ∈ [0, 27])
    early_vle    = vle_log[vle_log["date"] < 28]
    early_clicks = (early_vle.groupby("id_student")["sum_click"]
                              .sum().reset_index()
                              .rename(columns={"sum_click": "early_clicks"}))

    feat_df = info_dedup[["id_student",
                            "num_of_prev_attempts", "studied_credits"]].copy()
    feat_df = feat_df.merge(early_clicks, on="id_student", how="left").fillna(0)
    feat_df["early_clicks"] = np.log1p(feat_df["early_clicks"])

    num_arr  = feat_df[["num_of_prev_attempts", "studied_credits",
                         "early_clicks"]].values.astype(np.float32)
    raw_feat = np.concatenate([num_arr, cat_dummies], axis=1).astype(np.float32)
    return StandardScaler().fit_transform(raw_feat).astype(np.float32)


def _load_labels(raw_dir: str = "data/raw/oulad") -> Dict[str, torch.Tensor]:
    """Load grade/dropout/engagement labels from studentInfo.csv only (fast)."""
    raw = pathlib.Path(raw_dir)
    info = pd.read_csv(raw / "studentInfo.csv")
    info_dedup = (info.sort_values("code_presentation", ascending=False)
                      .drop_duplicates(subset="id_student")
                      .reset_index(drop=True))
    grade_map   = {"Distinction": 4.0, "Pass": 3.0, "Fail": 1.0, "Withdrawn": 0.0}
    dropout_map = {"Withdrawn": 1,     "Fail": 0,   "Pass": 0,   "Distinction": 0}
    engage_map  = {"Distinction": 2,   "Pass": 1,   "Fail": 0,   "Withdrawn": 0}
    fr = info_dedup["final_result"].fillna("Fail")
    return {
        "grade":      torch.tensor([grade_map.get(r, 1.0) for r in fr], dtype=torch.float),
        "dropout":    torch.tensor([dropout_map.get(r, 0)  for r in fr], dtype=torch.float),
        "engagement": torch.tensor([engage_map.get(r, 0)   for r in fr], dtype=torch.long),
    }


def _fast_student_features(data: HeteroData, text_proj: np.ndarray) -> np.ndarray:
    """
    Flatten student modalities without BERT (used by GAT, HAN):
      x_struct (N,64) + mean-pooled x_behav (N,32) + projected input_ids (N,32)
    → (N, 128) float32 array
    """
    x_struct = data["student"].x_struct.numpy()
    x_behav  = data["student"].x_behav.numpy().mean(axis=1)
    x_text   = data["student"].input_ids.float().numpy() @ text_proj
    return np.concatenate([x_struct, x_behav, x_text], axis=1).astype(np.float32)


def _make_bce(data: HeteroData, idx_t: torch.Tensor):
    """Weighted BCE closure; pos_weight = n_neg / n_pos on the training split."""
    y = data["student"].y_dropout[idx_t].float()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pw = torch.tensor([n_neg / max(n_pos, 1)])

    def _bce(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        true = true.float()
        w = torch.where(true.bool(), pw.expand_as(true), torch.ones_like(true))
        return F.binary_cross_entropy(pred, true, weight=w)

    return _bce


def _metapath_edges(
    src_dst_ei: torch.Tensor,
    n_src:      int,
    n_mid:      int,
) -> torch.Tensor:
    """
    Student-student meta-path edges via S→X→S (sparse matmul):
        A = incidence(student, mid_node)
        M = A @ A^T  (student × student co-occurrence)
    Self-loops removed. Falls back to identity self-loops if graph is empty.
    """
    row  = src_dst_ei[0].numpy()
    col  = src_dst_ei[1].numpy()
    ones = np.ones(len(row), dtype=np.float32)
    A    = sp.csr_matrix((ones, (row, col)), shape=(n_src, n_mid))
    M    = (A @ A.T).tocoo()
    M.setdiag(0)
    M.eliminate_zeros()

    if M.nnz == 0:
        idx = torch.arange(n_src, dtype=torch.long)
        return torch.stack([idx, idx])

    return torch.stack([
        torch.tensor(M.row.astype(np.int64)),
        torch.tensor(M.col.astype(np.int64)),
    ])


def _train_neural(
    model:        nn.Module,
    forward_fn,
    data:         HeteroData,
    train_idx:    np.ndarray,
    lambda_grade: float = _LAMBDA_GRADE,
    lambda_drop:  float = _LAMBDA_DROP,
    lambda_eng:   float = _LAMBDA_ENG,
    lr:           float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
    epochs:       int   = _TRAIN_EPOCHS,
) -> None:
    """Generic end-to-end training loop (used by GAT)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    ce_fn  = nn.CrossEntropyLoss()
    idx_t  = torch.tensor(train_idx, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = forward_fn()
        g = preds["grade"][idx_t].squeeze()
        d = preds["dropout"][idx_t].squeeze()
        e = preds["engagement"][idx_t]
        loss = (
            lambda_grade * mse_fn(g, data["student"].y_grade[idx_t])
            + lambda_drop  * bce_fn(d, data["student"].y_dropout[idx_t].float())
            + lambda_eng   * ce_fn(e, data["student"].y_engagement[idx_t])
        )
        loss.backward()
        optimizer.step()


# ----------------------------------------------------------------
# Prediction heads module shared by cached-embedding baselines
# ----------------------------------------------------------------

class _PredHeads(nn.Module):
    """Lightweight prediction heads used after node embeddings are cached."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.grade      = nn.Linear(d, 1)
        self.dropout_h  = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.engagement = nn.Linear(d, 3)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "grade":      self.grade(h),
            "dropout":    self.dropout_h(h),
            "engagement": self.engagement(h),
        }


def _evaluate_heads(
    heads: _PredHeads,
    h_all: torch.Tensor,
    idx_t: torch.Tensor,
    data:  HeteroData,
    mse_fn: nn.MSELoss,
    bce_fn,
    ce_fn:  nn.CrossEntropyLoss,
) -> Dict[str, float]:
    """Mirrors train_fast.py's evaluate_heads(), adapted to the _PredHeads /
    cached-embedding interface used by the HGT and HAN baselines."""
    heads.eval()
    with torch.no_grad():
        h   = h_all[idx_t]
        out = heads(h)
        val_loss = (
            _LAMBDA_GRADE * mse_fn(out["grade"].squeeze(),   data["student"].y_grade[idx_t])
            + _LAMBDA_DROP * bce_fn(out["dropout"].squeeze(), data["student"].y_dropout[idx_t])
            + _LAMBDA_ENG  * ce_fn(out["engagement"],         data["student"].y_engagement[idx_t])
        )

        g_pred = out["grade"].squeeze().numpy()
        g_true = data["student"].y_grade[idx_t].numpy()
        rmse   = float(np.sqrt(mean_squared_error(g_true, g_pred)))

        d_pred = out["dropout"].squeeze().numpy()
        d_true = data["student"].y_dropout[idx_t].numpy().astype(int)
        try:
            auc = float(roc_auc_score(d_true, d_pred))
        except ValueError:
            auc = 0.5

        e_pred = out["engagement"].argmax(dim=-1).numpy()
        e_true = data["student"].y_engagement[idx_t].numpy()
        f1 = float(f1_score(e_true, e_pred, average="macro", zero_division=0))

    return {"val_loss": float(val_loss), "val_rmse": rmse, "val_auc": auc, "val_f1": f1}


def _train_heads(
    heads:     _PredHeads,
    h_all:     torch.Tensor,
    data:      HeteroData,
    train_idx: np.ndarray,
    epochs:    int   = _TRAIN_HEAD_EPOCHS,
    lr:        float = _LR,
    patience:  int   = _HEAD_PATIENCE,
) -> None:
    """Train prediction heads on frozen cached node embeddings, with the
    same dual (AUC + F1) patience early-stopping rule as train_fast.py's
    train_one_fold_fast(): both metrics must stagnate for `patience`
    epochs before stopping. val_loss is tracked separately only to select
    the checkpoint restored at the end — it does not drive stopping,
    matching train_fast.py exactly.

    A held-out validation split is carved out of train_idx (stratified on
    the dropout label, 80/20, same random_state=42 as the rest of the
    codebase) so early stopping is not decided on data the model is
    ultimately scored on.
    """
    train_idx = np.asarray(train_idx)
    drop_labels = data["student"].y_dropout[train_idx].numpy()
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=_HEAD_VAL_FRAC, random_state=42)
    sub_train, sub_val = next(
        sss.split(np.zeros(len(train_idx)), drop_labels))
    inner_train_idx = torch.tensor(train_idx[sub_train], dtype=torch.long)
    inner_val_idx   = torch.tensor(train_idx[sub_val],   dtype=torch.long)

    optimizer = torch.optim.Adam(heads.parameters(), lr=lr)
    mse_fn    = nn.MSELoss()
    ce_fn     = nn.CrossEntropyLoss()
    bce_fn    = _make_bce(data, inner_train_idx)

    h_train = h_all[inner_train_idx].detach()

    best_val_loss = float("inf")
    best_val_auc  = -1.0
    best_val_f1   = -1.0
    auc_patience  = 0
    f1_patience   = 0
    best_state    = {k: v.clone() for k, v in heads.state_dict().items()}

    for epoch in range(epochs):
        heads.train()
        optimizer.zero_grad()
        p    = heads(h_train)
        loss = (
            _LAMBDA_GRADE * mse_fn(p["grade"].squeeze(), data["student"].y_grade[inner_train_idx])
            + _LAMBDA_DROP * bce_fn(p["dropout"].squeeze(), data["student"].y_dropout[inner_train_idx])
            + _LAMBDA_ENG  * ce_fn(p["engagement"],         data["student"].y_engagement[inner_train_idx])
        )
        loss.backward()
        optimizer.step()

        vm       = _evaluate_heads(heads, h_all, inner_val_idx, data,
                                    mse_fn, bce_fn, ce_fn)
        val_loss = vm["val_loss"]
        val_auc  = vm["val_auc"]
        val_f1   = vm["val_f1"]

        # Checkpoint on val_loss improvement (matches train_fast.py — this
        # tracks best state but does not itself drive early stopping)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in heads.state_dict().items()}

        # Dual-patience: both AUC and F1 must stagnate to trigger stop
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            auc_patience = 0
        else:
            auc_patience += 1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            f1_patience = 0
        else:
            f1_patience += 1

        if auc_patience >= patience and f1_patience >= patience:
            print(f"    Early stop at epoch {epoch + 1} "
                  f"(auc_pat={auc_patience}, f1_pat={f1_patience})")
            break

    # Restore best (lowest val_loss) checkpoint before returning
    heads.load_state_dict(best_state)


# ================================================================
# 1. XGBoostBaseline  (early-semester features — no leakage)
# ================================================================

class XGBoostBaseline:
    """
    One XGBRegressor (grade) + two XGBClassifiers (dropout, engagement).

    Uses ONLY early-semester features available before week 4:
      demographics + log(VLE clicks in days 0-27).
    Full-semester aggregates (total_clicks, mean_score, is_unregistered, etc.)
    are excluded to prevent data leakage into the dropout label.

    Both fit() and predict() are executed in a torch-free subprocess to avoid
    macOS OpenMP conflicts between PyTorch's libomp and XGBoost's libomp.
    Predictions for all N students are computed in one subprocess call and
    stored as numpy arrays; predict() just indexes into them.
    """

    def __init__(self, raw_dir: str = "data/raw/oulad") -> None:
        self._raw_dir         = raw_dir
        self._pred_grade:      Optional[np.ndarray] = None
        self._pred_dropout:    Optional[np.ndarray] = None
        self._pred_engagement: Optional[np.ndarray] = None

    def fit(self, data: HeteroData, train_idx: np.ndarray) -> None:
        import subprocess, sys

        X_all = _early_student_features(self._raw_dir)

        # Paths shared between parent and subprocess
        _x   = "/tmp/xgb_X.npy"
        _yg  = "/tmp/xgb_ygrade.npy"
        _yd  = "/tmp/xgb_ydrop.npy"
        _ye  = "/tmp/xgb_yeng.npy"
        _idx = "/tmp/xgb_train_idx.npy"
        _pg  = "/tmp/xgb_pred_grade.npy"
        _pd  = "/tmp/xgb_pred_drop.npy"
        _pe  = "/tmp/xgb_pred_eng.npy"

        np.save(_x,   X_all)
        np.save(_yg,  data["student"].y_grade.numpy())
        np.save(_yd,  data["student"].y_dropout.numpy().astype(int))
        np.save(_ye,  data["student"].y_engagement.numpy())
        np.save(_idx, train_idx)

        # Subprocess: fit on train_idx, predict for ALL students → numpy files
        # No torch import → no libomp conflict.
        script = (
            "import numpy as np\n"
            "from xgboost import XGBRegressor, XGBClassifier\n"
            f"Xa=np.load('{_x}');yg=np.load('{_yg}')\n"
            f"yd=np.load('{_yd}');ye=np.load('{_ye}')\n"
            f"idx=np.load('{_idx}')\n"
            "Xtr=Xa[idx]\n"
            "gm=XGBRegressor(n_estimators=100,max_depth=6,learning_rate=0.1,"
            "random_state=42,nthread=1)\n"
            "dm=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,"
            "random_state=42,nthread=1)\n"
            "em=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,"
            "random_state=42,nthread=1)\n"
            "gm.fit(Xtr,yg[idx]);dm.fit(Xtr,yd[idx]);em.fit(Xtr,ye[idx])\n"
            f"np.save('{_pg}',gm.predict(Xa))\n"
            f"np.save('{_pd}',dm.predict_proba(Xa)[:,1])\n"
            f"np.save('{_pe}',em.predict(Xa))\n"
        )
        res = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            raise RuntimeError(f"XGBoost subprocess failed:\n{res.stderr}")
        if res.stderr:
            print(f"XGBoost stderr: {res.stderr[:200]}", flush=True)

        # Load predictions into parent process — no XGBoost C-extension involved
        self._pred_grade      = np.load(_pg)
        self._pred_dropout    = np.load(_pd)
        self._pred_engagement = np.load(_pe)

    def predict(self, data: HeteroData, test_idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "grade":      self._pred_grade[test_idx],
            "dropout":    self._pred_dropout[test_idx],
            "engagement": self._pred_engagement[test_idx].astype(int),
        }


# ================================================================
# 2. GATBaseline
# ================================================================

class _GATNet(nn.Module):
    """2-layer GAT on homogeneous student-student co-enrollment graph."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, heads: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        head_dim = hidden_dim // heads
        self.conv1 = GATConv(in_dim,     head_dim,    heads=heads, concat=True,  dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim,  heads=1,     concat=False, dropout=dropout)
        self.drop  = nn.Dropout(dropout)
        self.grade_head      = nn.Linear(hidden_dim, 1)
        self.dropout_head    = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.engagement_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.drop(F.elu(self.conv1(x, edge_index)))
        x = self.drop(F.elu(self.conv2(x, edge_index)))
        return {
            "grade":      self.grade_head(x),
            "dropout":    self.dropout_head(x),
            "engagement": self.engagement_head(x),
        }


class GATBaseline:
    """
    Homogeneous student-student GAT where edges come from course co-enrollment.
    No BERT; random projection used for text proxy.
    """

    def __init__(self, hidden_dim: int = 128, heads: int = 4,
                 dropout: float = 0.3) -> None:
        np.random.seed(42)
        self._text_proj  = (np.random.randn(32, 32) / np.sqrt(32)).astype(np.float32)
        self._hidden_dim = hidden_dim
        self._heads      = heads
        self._dropout    = dropout
        self._model:      Optional[_GATNet]      = None
        self._edge_index: Optional[torch.Tensor] = None

    def fit(self, data: HeteroData, train_idx: np.ndarray) -> None:
        n_s = data["student"].x_struct.shape[0]
        n_c = data["course"].input_ids.shape[0]

        ei_sc = data["student", "enrolled_in", "course"].edge_index
        row, col = ei_sc[0].numpy(), ei_sc[1].numpy()
        ones = np.ones(len(row), dtype=np.float32)
        A = sp.csr_matrix((ones, (row, col)), shape=(n_s, n_c))
        SCS = (A @ A.T).tocoo()
        SCS.setdiag(0)
        SCS.eliminate_zeros()
        self._edge_index = torch.stack([
            torch.tensor(SCS.row.astype(np.int64)),
            torch.tensor(SCS.col.astype(np.int64)),
        ])

        X_all = _fast_student_features(data, self._text_proj)
        x_t   = torch.tensor(X_all, dtype=torch.float32)
        self._model = _GATNet(X_all.shape[1], self._hidden_dim, self._heads, self._dropout)

        _train_neural(
            self._model,
            lambda: self._model(x_t, self._edge_index),
            data, train_idx,
        )

    def predict(self, data: HeteroData, test_idx: np.ndarray) -> Dict[str, np.ndarray]:
        X_all = _fast_student_features(data, self._text_proj)
        x_t   = torch.tensor(X_all, dtype=torch.float32)
        idx_t = torch.tensor(test_idx, dtype=torch.long)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(x_t, self._edge_index)
        return {
            "grade":      preds["grade"][idx_t].squeeze().numpy(),
            "dropout":    preds["dropout"][idx_t].squeeze().numpy(),
            "engagement": preds["engagement"][idx_t].argmax(dim=-1).numpy(),
        }


# ================================================================
# 3. HGTBaseline  (key ablation: same compute as MG_HGNN, no gate)
# ================================================================

class _HGTNet(nn.Module):
    """
    MG_HGNN without the ModalityGate:
      [h_struct || h_text || h_behav] → Linear(3d→d) → HGTConv × 2

    Exposes embed() for the fast-training strategy:
      1. Warm up with forward_cached() for a few full-graph epochs.
      2. Cache embeddings via embed() under no_grad.
      3. Train prediction heads only on the cached tensor.
    """

    def __init__(self, cfg: Config, build_text_encoder: bool = True) -> None:
        """
        build_text_encoder=False skips loading the real BERT-backed
        TextEncoder at construction time. embed()/forward_cached() never
        touch self.text_enc directly (they consume a precomputed
        `cached_text` dict), so when a valid BERT embedding cache already
        exists on disk this avoids loading a full BertModel (~440 MB) into
        memory for nothing — call ensure_text_enc() first if you later
        need self.text_enc for real (e.g. the cache-miss fallback in
        HGTBaseline._get_bert_cache).
        """
        super().__init__()
        d = cfg.embed_dim
        self._cfg = cfg

        self.struct_enc = StructuredEncoder(cfg)
        self.text_enc   = TextEncoder(cfg, freeze_bert=True) if build_text_encoder else None
        self.behav_enc  = BehavioralEncoder(cfg)
        self.inst_enc   = nn.Sequential(
            nn.Linear(cfg.structured_input_dim, d),
            nn.LayerNorm(d), nn.ReLU(), nn.Dropout(cfg.dropout),
        )

        self.modality_proj = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        metadata = (cfg.node_types, _EDGE_TRIPLES)
        self.convs = nn.ModuleList([
            HGTConv(in_channels=d, out_channels=d,
                    metadata=metadata, heads=cfg.num_heads)
            for _ in range(cfg.num_hgnn_layers)
        ])
        self.conv_drop = nn.Dropout(cfg.dropout)

        self.grade_head      = nn.Linear(d, 1)
        self.dropout_head    = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.engagement_head = nn.Linear(d, 3)

    def ensure_text_enc(self) -> TextEncoder:
        """Build the real BERT-backed TextEncoder on first use, if it
        wasn't constructed eagerly (build_text_encoder=False)."""
        if self.text_enc is None:
            self.text_enc = TextEncoder(self._cfg, freeze_bert=True)
        return self.text_enc

    def embed(
        self,
        data:        HeteroData,
        cached_text: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run encoders + HGTConv; return student embeddings before heads."""
        h_s = self.struct_enc(data["student"].x_struct)
        h_b = self.behav_enc(data["student"].x_behav)
        x_student = self.modality_proj(
            torch.cat([h_s, cached_text["student"], h_b], dim=-1)
        )
        x_dict = {
            "student":    x_student,
            "course":     cached_text["course"],
            "resource":   cached_text["resource"],
            "instructor": self.inst_enc(data["instructor"].x_struct),
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.conv_drop(F.relu(v)) for k, v in x_dict.items()}
        return x_dict["student"]

    def forward_cached(
        self,
        data:        HeteroData,
        cached_text: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        s = self.embed(data, cached_text)
        return {
            "grade":      self.grade_head(s),
            "dropout":    self.dropout_head(s),
            "engagement": self.engagement_head(s),
        }


class HGTBaseline:
    """
    Same HGTConv backbone as MG_HGNN; ablates the ModalityGate.

    Fast-training strategy (matches MG-HGNN ablation protocol):
      1. Pre-load BERT embeddings from cache (avoids re-running BERT each epoch).
      2. Warm up: _HGT_WARMUP_EPOCHS full-graph epochs so HGTConv learns
         meaningful node representations.
      3. Cache student embeddings via embed() under no_grad.
      4. Train prediction heads on the cached tensor for up to
         _TRAIN_HEAD_EPOCHS epochs (class-weighted BCE for dropout),
         with dual (AUC + F1) patience early stopping on a val split
         carved from train_idx — same rule as train_fast.py.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self._cfg          = cfg or Config()
        self._model:  Optional[_HGTNet]             = None
        self._h_all:  Optional[torch.Tensor]         = None
        self._heads:  Optional[_PredHeads]           = None

    def _try_load_bert_cache(self, data: HeteroData) -> Optional[Dict[str, torch.Tensor]]:
        """Attempt to load pre-computed BERT embeddings from disk. Returns
        None on a miss (missing file or shape mismatch) — deliberately
        needs no model, so it can run before _HGTNet (and therefore
        before any real BertModel) is constructed."""
        try:
            raw = torch.load(_BERT_CACHE_PATH, weights_only=True)
            if all(
                ntype in raw and raw[ntype].shape[0] == data[ntype].num_nodes
                for ntype in ("student", "course", "resource")
                if ntype in data.node_types
            ):
                return {k: v.detach() for k, v in raw.items()}
        except FileNotFoundError:
            pass
        return None

    def _compute_bert_cache_fallback(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Run real BERT to build the embedding cache (broadcast or batched).
        Only called on a cache miss; builds self._model.text_enc lazily."""
        text_enc = self._model.ensure_text_enc()

        # Detect text-free datasets: all nodes have [CLS]-only stubs
        # (input_ids[:,0]==101, rest zero; mask[:,0]==1, rest zero).
        # One BERT call is sufficient — broadcast to every node type.
        ids  = data["student"].input_ids
        mask = data["student"].attention_mask
        is_cls_only = (
            (ids[:, 0] == 101).all()
            and (ids[:, 1:] == 0).all()
            and (mask[:, 0] == 1).all()
            and (mask[:, 1:] == 0).all()
        )
        if is_cls_only:
            with torch.no_grad():
                emb = text_enc(ids[:1], mask[:1])  # (1, d)
            return {
                ntype: emb.expand(data[ntype].num_nodes, -1).clone()
                for ntype in ("student", "course", "resource")
                if ntype in data.node_types
            }

        # Batched fallback (256 rows at a time to avoid OOM)
        result: Dict[str, torch.Tensor] = {}
        for ntype in ("student", "course", "resource"):
            ids  = data[ntype].input_ids
            mask = data[ntype].attention_mask
            chunks = []
            with torch.no_grad():
                for i in range(0, len(ids), 256):
                    h = text_enc(ids[i:i+256], mask[i:i+256])
                    chunks.append(h.detach())
            result[ntype] = torch.cat(chunks, dim=0)
        return result

    def _get_bert_cache(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Load pre-computed BERT embeddings; use broadcast or batched
        fallback (real BERT) only on a cache miss."""
        cached = self._try_load_bert_cache(data)
        if cached is not None:
            return cached
        return self._compute_bert_cache_fallback(data)

    def fit(self, data: HeteroData, train_idx: np.ndarray) -> None:
        cfg           = self._cfg
        # Check the on-disk cache BEFORE building _HGTNet, so a real
        # BertModel (~440 MB) is only constructed on an actual cache miss.
        cached_text   = self._try_load_bert_cache(data)
        self._model   = _HGTNet(cfg, build_text_encoder=(cached_text is None))
        if cached_text is None:
            cached_text = self._compute_bert_cache_fallback(data)

        # Warmup: train full model so HGTConv weights are meaningful
        opt   = torch.optim.Adam(self._model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
        idx_t = torch.tensor(train_idx, dtype=torch.long)
        mse_fn, ce_fn = nn.MSELoss(), nn.CrossEntropyLoss()
        bce_fn = _make_bce(data, idx_t)

        self._model.train()
        for _ in range(_HGT_WARMUP_EPOCHS):
            opt.zero_grad()
            p = self._model.forward_cached(data, cached_text)
            loss = (
                _LAMBDA_GRADE * mse_fn(p["grade"][idx_t].squeeze(),    data["student"].y_grade[idx_t])
                + _LAMBDA_DROP * bce_fn(p["dropout"][idx_t].squeeze(),   data["student"].y_dropout[idx_t])
                + _LAMBDA_ENG  * ce_fn(p["engagement"][idx_t],           data["student"].y_engagement[idx_t])
            )
            loss.backward()
            opt.step()

        # Cache student embeddings from the warmed-up model
        self._model.eval()
        with torch.no_grad():
            self._h_all = self._model.embed(data, cached_text)

        # Head-only training on cached embeddings
        d             = self._h_all.shape[1]
        self._heads   = _PredHeads(d)
        _train_heads(self._heads, self._h_all, data, train_idx)

    def predict(self, data: HeteroData, test_idx: np.ndarray) -> Dict[str, np.ndarray]:
        idx_t = torch.tensor(test_idx, dtype=torch.long)
        self._heads.eval()
        with torch.no_grad():
            p = self._heads(self._h_all[idx_t])
        return {
            "grade":      p["grade"].squeeze().numpy(),
            "dropout":    p["dropout"].squeeze().numpy(),
            "engagement": p["engagement"].argmax(dim=-1).numpy(),
        }


# ================================================================
# 4. HANBaseline
# ================================================================

class _HANNet(nn.Module):
    """
    2-layer HANConv with two semantic meta-paths:
      SCS: student → course → student   (co-enrollment)
      SRS: student → resource → student  (co-access, edge-capped)
    """

    _METADATA = (
        ["student"],
        [("student", "SCS", "student"), ("student", "SRS", "student")],
    )

    def __init__(self, in_dim: int, hidden_dim: int = 128, heads: int = 1,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = HANConv(in_dim,     hidden_dim, metadata=self._METADATA,
                             heads=heads, dropout=dropout)
        self.conv2 = HANConv(hidden_dim, hidden_dim, metadata=self._METADATA,
                             heads=heads, dropout=dropout)
        self.drop  = nn.Dropout(dropout)
        self.grade_head      = nn.Linear(hidden_dim, 1)
        self.dropout_head    = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.engagement_head = nn.Linear(hidden_dim, 3)

    def embed(
        self,
        x:      torch.Tensor,
        scs_ei: torch.Tensor,
        srs_ei: torch.Tensor,
    ) -> torch.Tensor:
        """Run HANConv × 2; return student embeddings before heads."""
        x_dict  = {"student": x}
        ei_dict = {
            ("student", "SCS", "student"): scs_ei,
            ("student", "SRS", "student"): srs_ei,
        }
        x_dict = self.conv1(x_dict, ei_dict)
        x_dict = {k: self.drop(F.relu(v)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, ei_dict)
        x_dict = {k: self.drop(F.relu(v)) for k, v in x_dict.items()}
        return x_dict["student"]

    def forward(
        self,
        x:      torch.Tensor,
        scs_ei: torch.Tensor,
        srs_ei: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h = self.embed(x, scs_ei, srs_ei)
        return {
            "grade":      self.grade_head(h),
            "dropout":    self.dropout_head(h),
            "engagement": self.engagement_head(h),
        }


class HANBaseline:
    """
    Two-meta-path HAN: SCS (student-course-student) and SRS (student-resource-student).

    Fast-training strategy:
      1. Cap accessed edges to _HAN_MAX_ACCESSED before building SRS to avoid
         O(N²) co-access density.
      2. Warm up: _HAN_WARMUP_EPOCHS full-graph epochs so HANConv weights are
         meaningful.
      3. Cache student embeddings via embed() under no_grad.
      4. Train prediction heads for up to _TRAIN_HEAD_EPOCHS epochs
         (class-weighted BCE), with the same dual-patience early
         stopping as HGTBaseline / train_fast.py.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        np.random.seed(42)
        self._text_proj  = (np.random.randn(32, 32) / np.sqrt(32)).astype(np.float32)
        self._hidden_dim = hidden_dim
        self._dropout    = dropout
        self._model:  Optional[_HANNet]      = None
        self._scs_ei: Optional[torch.Tensor] = None
        self._srs_ei: Optional[torch.Tensor] = None
        self._h_all:  Optional[torch.Tensor] = None
        self._heads:  Optional[_PredHeads]   = None

    def fit(self, data: HeteroData, train_idx: np.ndarray) -> None:
        n_s = data["student"].x_struct.shape[0]
        n_c = data["course"].input_ids.shape[0]
        n_r = data["resource"].input_ids.shape[0]

        # SCS from enrolled_in (already capped by run_all_baselines)
        self._scs_ei = _metapath_edges(
            data["student", "enrolled_in", "course"].edge_index, n_s, n_c
        )

        # SRS: cap accessed before matmul to keep co-access graph sparse
        acc_ei = data["student", "accessed", "resource"].edge_index
        if acc_ei.shape[1] > _HAN_MAX_ACCESSED:
            perm   = torch.randperm(acc_ei.shape[1])[:_HAN_MAX_ACCESSED]
            acc_ei = acc_ei[:, perm]
        self._srs_ei = _metapath_edges(acc_ei, n_s, n_r)

        X_all = _fast_student_features(data, self._text_proj)
        x_t   = torch.tensor(X_all, dtype=torch.float32)
        self._model = _HANNet(X_all.shape[1], self._hidden_dim, heads=1,
                              dropout=self._dropout)

        # Warmup: train full model for a few epochs
        scs, srs = self._scs_ei, self._srs_ei
        opt   = torch.optim.Adam(self._model.parameters(), lr=_LR,
                                  weight_decay=_WEIGHT_DECAY)
        idx_t = torch.tensor(train_idx, dtype=torch.long)
        mse_fn, ce_fn = nn.MSELoss(), nn.CrossEntropyLoss()
        bce_fn = _make_bce(data, idx_t)

        self._model.train()
        for _ in range(_HAN_WARMUP_EPOCHS):
            opt.zero_grad()
            p = self._model(x_t, scs, srs)
            loss = (
                _LAMBDA_GRADE * mse_fn(p["grade"][idx_t].squeeze(),    data["student"].y_grade[idx_t])
                + _LAMBDA_DROP * bce_fn(p["dropout"][idx_t].squeeze(),   data["student"].y_dropout[idx_t])
                + _LAMBDA_ENG  * ce_fn(p["engagement"][idx_t],           data["student"].y_engagement[idx_t])
            )
            loss.backward()
            opt.step()

        # Cache student embeddings
        self._model.eval()
        with torch.no_grad():
            self._h_all = self._model.embed(x_t, scs, srs)

        # Head-only training
        self._heads = _PredHeads(self._hidden_dim)
        _train_heads(self._heads, self._h_all, data, train_idx)

    def predict(self, data: HeteroData, test_idx: np.ndarray) -> Dict[str, np.ndarray]:
        idx_t = torch.tensor(test_idx, dtype=torch.long)
        self._heads.eval()
        with torch.no_grad():
            p = self._heads(self._h_all[idx_t])
        return {
            "grade":      p["grade"].squeeze().numpy(),
            "dropout":    p["dropout"].squeeze().numpy(),
            "engagement": p["engagement"].argmax(dim=-1).numpy(),
        }


# ================================================================
# Comparison runner
# ================================================================

def run_all_baselines(
    data:      HeteroData,
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    cfg:       Optional[Config] = None,
    raw_dir:   str = "data/raw/oulad",
) -> Dict[str, Dict[str, float]]:
    """
    Fit and score all four baselines, then print a comparison table.
    Returns nested dict: {model_name: {rmse, auc, f1}}.

    Labels are auto-loaded from studentInfo.csv if not already set on data.
    Edge caps (accessed → 100 K, enrolled_in → 2 K) are applied in-place
    to keep HGTConv and SCS computation tractable on CPU.
    """
    cfg = cfg or Config()

    # Auto-load labels if the caller hasn't attached them
    if not hasattr(data["student"], "y_grade"):
        labels = _load_labels(raw_dir)
        data["student"].y_grade      = labels["grade"]
        data["student"].y_dropout    = labels["dropout"]
        data["student"].y_engagement = labels["engagement"]

    # Edge caps (in-place) — keep HGTConv and co-enrollment SCS tractable
    _acc_et = ("student", "accessed", "resource")
    _acc_ei = data[_acc_et].edge_index
    if _acc_ei.shape[1] > 100_000:
        perm = torch.randperm(_acc_ei.shape[1])[:100_000]
        data[_acc_et].edge_index = _acc_ei[:, perm]

    _enr_et = ("student", "enrolled_in", "course")
    _enr_ei = data[_enr_et].edge_index
    if _enr_ei.shape[1] > 2_000:
        perm = torch.randperm(_enr_ei.shape[1])[:2_000]
        data[_enr_et].edge_index = _enr_ei[:, perm]
    print(f"[cap] enrolled_in: {_enr_ei.shape[1]} -> "
          f"{data[_enr_et].edge_index.shape[1]}", flush=True)

    baselines: Dict[str, object] = {
        "XGBoost": XGBoostBaseline(raw_dir),
        "GAT":     GATBaseline(),
        "HGT":     HGTBaseline(cfg),
        "HAN":     HANBaseline(),
    }

    y_grade = data["student"].y_grade[test_idx].numpy()
    y_drop  = data["student"].y_dropout[test_idx].numpy().astype(int)
    y_eng   = data["student"].y_engagement[test_idx].numpy()

    results: Dict[str, Dict[str, float]] = {}

    for name, model in baselines.items():
        print(f"  [{name}] fitting...", flush=True)
        model.fit(data, train_idx)
        preds = model.predict(data, test_idx)

        rmse = float(np.sqrt(mean_squared_error(y_grade, preds["grade"])))
        try:
            auc = float(roc_auc_score(y_drop, preds["dropout"]))
        except ValueError:
            auc = 0.5
        f1 = float(f1_score(y_eng, preds["engagement"],
                             average="macro", zero_division=0))

        results[name] = {"rmse": rmse, "auc": auc, "f1": f1}
        print(f"  [{name}] RMSE={rmse:.4f}  AUC={auc:.4f}  F1={f1:.4f}", flush=True)

    W = 54
    print(f"\n{'='*W}")
    print(f"  {'Model':<14}  {'RMSE':>8}   {'AUC-ROC':>8}   {'Eng-F1':>8}")
    print(f"{'='*W}")
    for name, m in results.items():
        print(f"  {name:<14}  {m['rmse']:>8.4f}   {m['auc']:>8.4f}   {m['f1']:>8.4f}")
    print(f"{'='*W}\n")

    return results


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold

    parser = argparse.ArgumentParser(description="Baseline comparison for MG-HGNN")
    parser.add_argument("--dataset",
                        choices=["oulad", "assistments09", "assistments15"],
                        default="oulad")
    args = parser.parse_args()

    cfg = Config()

    if args.dataset == "oulad":
        from data.oulad_loader import OULADLoader
        print("Loading OULAD...")
        data_loader = OULADLoader(cfg)
        data, meta  = data_loader.load()
        raw_dir     = "data/raw/oulad"
        skip_xgb    = False
    else:
        from data.assistments_loader import ASSISTmentsLoader
        ver = "2009" if args.dataset == "assistments09" else "2015"
        print(f"Loading ASSISTments {ver}...")
        data_loader = ASSISTmentsLoader(cfg, version=ver)
        if not data_loader.check_files():
            return
        data, meta = data_loader.load()
        raw_dir    = None
        skip_xgb   = True   # XGBoost uses OULAD-specific CSV features

    data["student"].y_grade      = meta["grade"].float()
    data["student"].y_dropout    = meta["dropout"].float()
    data["student"].y_engagement = meta["engagement"].long()

    n_students    = len(meta["student_ids"])
    dropout_labels = meta["dropout"].numpy()
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(n_students), dropout_labels))

    all_results: Dict[str, Dict] = {}
    Path("results").mkdir(exist_ok=True)

    for fold_k, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_k + 1}/5 ===")
        fold_res = run_all_baselines(
            data, train_idx, val_idx, cfg=cfg,
            raw_dir=raw_dir if raw_dir else "data/raw/oulad",
        )
        for model_name, metrics in fold_res.items():
            if skip_xgb and model_name == "XGBoost":
                continue
            if model_name not in all_results:
                all_results[model_name] = {"rmse": [], "auc": [], "f1": []}
            all_results[model_name]["rmse"].append(metrics["rmse"])
            all_results[model_name]["auc"].append(metrics["auc"])
            all_results[model_name]["f1"].append(metrics["f1"])

    summary: Dict[str, Dict] = {}
    W = 60
    print(f"\n{'='*W}")
    print(f"  Baseline Summary — {args.dataset.upper()}  (5-fold CV)")
    print(f"{'='*W}")
    print(f"  {'Model':<16} {'AUC':>8}  {'±':>6}  {'RMSE':>8}  {'F1':>8}")
    print(f"  {'-'*W}")
    for name, vals in all_results.items():
        auc_m  = float(np.mean(vals["auc"]))
        auc_s  = float(np.std(vals["auc"]))
        rmse_m = float(np.mean(vals["rmse"]))
        f1_m   = float(np.mean(vals["f1"]))
        summary[name] = {"auc_mean": auc_m, "auc_std": auc_s,
                         "rmse_mean": rmse_m, "f1_mean": f1_m}
        print(f"  {name:<16} {auc_m:>8.4f}  {auc_s:>6.4f}  "
              f"{rmse_m:>8.4f}  {f1_m:>8.4f}")
    print(f"{'='*W}")

    out_path = Path("results") / f"baseline_results_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
