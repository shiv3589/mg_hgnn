"""Baseline models for MG-HGNN comparison experiments.

Four baselines share a unified two-method interface::

    baseline.fit(data, labels, train_mask, val_mask=None)
    preds = baseline.predict(data)
    # preds["grade"]      : (N,) float scores
    # preds["dropout"]    : (N,) sigmoid probabilities in [0, 1]
    # preds["engagement"] : (N, 3) class logits

Usage::

    from baselines import XGBoostBaseline, GATBaseline, HGTBaseline, HANBaseline
    from train import MultiTaskLabels   # or any duck-typed labels object

    for bl in [XGBoostBaseline(), GATBaseline(), HGTBaseline(), HANBaseline()]:
        bl.fit(data, labels, train_mask, val_mask)
        preds = bl.predict(data)
        grade_metrics = evaluate(preds["grade"][val_mask], labels.grade[val_mask], "grade")

Baselines
---------
XGBoostBaseline  — flat multi-modal features; three XGB models; no graph structure
GATBaseline      — student-student co-enrollment graph; 2-layer GAT
HGTBaseline      — full heterogeneous graph; 2-layer HGT; no modality gate
HANBaseline      — SCS and SRS meta-paths; 2-layer HAN with semantic attention
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HGTConv
from tqdm import tqdm

try:
    from torch_geometric.nn import HANConv as _HANConv
    _HAS_HANCONV = True
except ImportError:
    _HAS_HANCONV = False

try:
    import xgboost as _xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import scipy.sparse as _sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from models.encoders import BehavioralEncoder, StructuredEncoder


# ---------------------------------------------------------------------------
# Graph schema constants (mirrors MGHGNN)
# ---------------------------------------------------------------------------

_NODE_TYPES: List[str] = ["student", "course", "instructor", "resource"]
_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("student", "enrolled_in",       "course"),
    ("student", "collaborated_with", "student"),
    ("student", "accessed",          "resource"),
    ("student", "submitted_to",      "course"),
]
_HGNN_METADATA: Tuple = (_NODE_TYPES, _EDGE_TYPES)

# HAN uses only the two meta-path-forming edge types plus their reverses.
_HAN_NODE_TYPES: List[str] = ["student", "course", "resource"]
_HAN_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("student", "enrolled_in",      "course"),
    ("course",  "rev_enrolled_in",  "student"),
    ("student", "accessed",         "resource"),
    ("resource", "rev_accessed",    "student"),
]
_HAN_METADATA: Tuple = (_HAN_NODE_TYPES, _HAN_EDGE_TYPES)


# ---------------------------------------------------------------------------
# Standalone multi-task loss
# ---------------------------------------------------------------------------


def _multitask_loss(
    preds: Dict[str, Tensor],
    grade: Tensor,
    dropout: Tensor,
    engagement: Tensor,
    mask: Tensor,
    lw_grade: float = 0.4,
    lw_dropout: float = 0.4,
    lw_engagement: float = 0.2,
) -> Tensor:
    """Weighted multi-task loss over masked student nodes.

    Args:
        preds: Forward dict with "grade" (N,), "dropout" (N,) *raw logits*,
            "engagement" (N, 3) *raw logits*.
        grade, dropout, engagement: Label tensors of shape (N,).
        mask: Boolean mask selecting the students to include.
        lw_*: Scalar weights (should sum to 1 for clean bookkeeping).

    Returns:
        Scalar weighted sum: lw_grade·MSE + lw_dropout·BCE + lw_engagement·CE.
    """
    dev = preds["grade"].device
    l_g = F.mse_loss(
        preds["grade"][mask],
        grade[mask].to(dev),
    )
    l_d = F.binary_cross_entropy_with_logits(
        preds["dropout"][mask],
        dropout[mask].to(dev),
    )
    l_e = F.cross_entropy(
        preds["engagement"][mask],
        engagement[mask].long().to(dev),
    )
    return lw_grade * l_g + lw_dropout * l_d + lw_engagement * l_e


# ---------------------------------------------------------------------------
# Lightweight prediction heads (raw logits; sigmoid applied in predict())
# ---------------------------------------------------------------------------


class _GradeHead(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)


class _DropoutHead(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)


class _EngagementHead(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseBaseline(abc.ABC):
    """Common interface for all MG-HGNN comparison baselines.

    The ``labels`` argument to :meth:`fit` is any object exposing
    ``.grade``, ``.dropout``, and ``.engagement`` tensor attributes
    (e.g. :class:`train.MultiTaskLabels`).
    """

    @abc.abstractmethod
    def fit(
        self,
        data: HeteroData,
        labels,
        train_mask: Tensor,
        val_mask: Optional[Tensor] = None,
    ) -> None:
        """Train on the student nodes selected by *train_mask*."""

    @abc.abstractmethod
    def predict(self, data: HeteroData) -> Dict[str, Tensor]:
        """Return predictions for **all** student nodes.

        Returns:
            ``"grade"``      — ``(N,)`` float regression scores.
            ``"dropout"``    — ``(N,)`` dropout probabilities in ``[0, 1]``.
            ``"engagement"`` — ``(N, 3)`` engagement class logits.
        """


# ---------------------------------------------------------------------------
# 1. XGBoostBaseline
# ---------------------------------------------------------------------------


class XGBoostBaseline(BaseBaseline):
    """Three independent XGBoost models — one per prediction task.

    No graph structure is used.  All student modality features are
    concatenated into a single flat vector:

    * ``data["student"].x``        — tabular structured features
    * ``data["student"].text_x``   — pre-computed BERT-size embeddings
    * ``data["student"].behavior`` — mean-pooled over the time axis to ``(N, F)``

    Args:
        n_estimators: Boosting rounds for each model.
        max_depth: Maximum tree depth.
        learning_rate: XGBoost ``eta`` parameter.
        tree_method: ``"hist"`` (CPU) or ``"gpu_hist"`` (GPU).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        tree_method: str = "hist",
    ) -> None:
        if not _HAS_XGB:
            raise ImportError(
                "XGBoostBaseline requires xgboost. "
                "Install with: pip install xgboost"
            )
        self._xgb_kw = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method=tree_method,
            verbosity=0,
        )
        self._grade_mdl: Optional[_xgb.XGBRegressor]    = None
        self._dropout_mdl: Optional[_xgb.XGBClassifier]  = None
        self._engage_mdl: Optional[_xgb.XGBClassifier]   = None

    # ------------------------------------------------------------------

    @staticmethod
    def _extract(data: HeteroData) -> np.ndarray:
        """Build flat feature matrix from all student modalities."""
        parts = [data["student"].x.cpu().float().numpy()]
        if hasattr(data["student"], "text_x") and data["student"].text_x is not None:
            parts.append(data["student"].text_x.cpu().float().numpy())
        if hasattr(data["student"], "behavior") and data["student"].behavior is not None:
            b = data["student"].behavior.cpu().float().numpy()  # (N, T, F)
            parts.append(b.mean(axis=1))                        # (N, F)
        return np.concatenate(parts, axis=1)

    def fit(
        self,
        data: HeteroData,
        labels,
        train_mask: Tensor,
        val_mask: Optional[Tensor] = None,
    ) -> None:
        X   = self._extract(data)
        idx = train_mask.cpu().numpy().astype(bool)

        y_grade  = labels.grade.cpu().float().numpy()[idx]
        y_drop   = labels.dropout.cpu().long().numpy()[idx]
        y_engage = labels.engagement.cpu().long().numpy()[idx]

        self._grade_mdl = _xgb.XGBRegressor(
            objective="reg:squarederror", **self._xgb_kw,
        )
        self._grade_mdl.fit(X[idx], y_grade)

        self._dropout_mdl = _xgb.XGBClassifier(
            objective="binary:logistic", **self._xgb_kw,
        )
        self._dropout_mdl.fit(X[idx], y_drop)

        self._engage_mdl = _xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, **self._xgb_kw,
        )
        self._engage_mdl.fit(X[idx], y_engage)

    def predict(self, data: HeteroData) -> Dict[str, Tensor]:
        if self._grade_mdl is None:
            raise RuntimeError("Call fit() before predict().")

        X = self._extract(data)

        grade = torch.tensor(
            self._grade_mdl.predict(X).astype(np.float32),
            dtype=torch.float,
        )
        # predict_proba → (N, 2); column 1 is P(dropout=1)
        dropout = torch.tensor(
            self._dropout_mdl.predict_proba(X)[:, 1].astype(np.float32),
            dtype=torch.float,
        )
        # multi:softprob predict_proba → (N, 3)
        engagement = torch.tensor(
            self._engage_mdl.predict_proba(X).astype(np.float32),
            dtype=torch.float,
        )
        return {"grade": grade, "dropout": dropout, "engagement": engagement}


# ---------------------------------------------------------------------------
# Shared neural training base
# ---------------------------------------------------------------------------


class _NeuralBaseline(nn.Module, BaseBaseline):
    """Abstract base for GNN baselines with a shared multi-task training loop.

    Subclasses implement :meth:`_build` (lazy module init from data dims) and
    :meth:`_forward_impl` (model-specific forward pass returning raw logits).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        epochs: int = 200,
        patience: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Union[str, torch.device] = "cpu",
        lambda_grade: float = 0.4,
        lambda_dropout: float = 0.4,
        lambda_engagement: float = 0.2,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_dim   = hidden_dim
        self.num_heads    = num_heads
        self.dropout_rate = dropout
        self._epochs      = epochs
        self._patience    = patience
        self._lr          = lr
        self._wd          = weight_decay
        self._device      = torch.device(device)
        self._lw_g        = lambda_grade
        self._lw_d        = lambda_dropout
        self._lw_e        = lambda_engagement
        self._built       = False

    # ---- Subclass contract ------------------------------------------------

    def _build(self, data: HeteroData) -> None:
        """Lazily initialise all nn.Module submodules from data dims."""
        raise NotImplementedError

    def _forward_impl(self, data: HeteroData) -> Dict[str, Tensor]:
        """Model-specific forward. Returns raw logits for dropout."""
        raise NotImplementedError

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        return self._forward_impl(data)

    # ---- Shared training loop ---------------------------------------------

    def fit(
        self,
        data: HeteroData,
        labels,
        train_mask: Tensor,
        val_mask: Optional[Tensor] = None,
    ) -> None:
        # Lazy build from data feature dims
        if not self._built:
            self._build(data)
            self._built = True

        self.to(self._device)
        data       = data.to(self._device)
        train_mask = train_mask.to(self._device)
        if val_mask is not None:
            val_mask = val_mask.to(self._device)

        grade_y      = labels.grade.to(self._device)
        dropout_y    = labels.dropout.to(self._device)
        engagement_y = labels.engagement.to(self._device)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._lr, weight_decay=self._wd
        )

        best_val   = float("inf")
        patience_n = 0
        best_state: Optional[Dict] = None

        pbar = tqdm(
            range(1, self._epochs + 1),
            desc=type(self).__name__,
            unit="ep",
            leave=True,
        )
        for epoch in pbar:
            # Train
            self.train()
            optimizer.zero_grad()
            preds = self._forward_impl(data)
            loss = _multitask_loss(
                preds, grade_y, dropout_y, engagement_y,
                train_mask, self._lw_g, self._lw_d, self._lw_e,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            post: Dict[str, str] = {"tr": f"{loss.item():.4f}"}

            # Validate + early stop
            if val_mask is not None:
                self.eval()
                with torch.no_grad():
                    preds_v = self._forward_impl(data)
                    vl = _multitask_loss(
                        preds_v, grade_y, dropout_y, engagement_y,
                        val_mask, self._lw_g, self._lw_d, self._lw_e,
                    ).item()

                post["val"] = f"{vl:.4f}"

                if vl < best_val:
                    best_val   = vl
                    patience_n = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_n += 1

                if patience_n >= self._patience:
                    pbar.close()
                    print(
                        f"  [{type(self).__name__}] early stop at epoch {epoch} "
                        f"(best val {best_val:.4f})"
                    )
                    break

            pbar.set_postfix(**post)

        if best_state is not None:
            self.load_state_dict(best_state)

    # ---- Predict ----------------------------------------------------------

    @torch.no_grad()
    def predict(self, data: HeteroData) -> Dict[str, Tensor]:
        if not self._built:
            raise RuntimeError(f"{type(self).__name__}.predict() called before fit().")
        self.eval()
        data  = data.to(self._device)
        preds = self._forward_impl(data)
        return {
            "grade":      preds["grade"].cpu(),
            "dropout":    torch.sigmoid(preds["dropout"]).cpu(),
            "engagement": preds["engagement"].cpu(),
        }

    # ---- Shared sub-module factories -------------------------------------

    def _text_proj(self, text_dim: int) -> nn.Sequential:
        d = self.hidden_dim
        return nn.Sequential(
            nn.Linear(text_dim, d),
            nn.LayerNorm(d),
            nn.Dropout(self.dropout_rate),
        )

    def _fuse_block(self, in_dim: int) -> nn.Sequential:
        """Project concatenated modality embeddings down to hidden_dim."""
        d = self.hidden_dim
        return nn.Sequential(
            nn.Linear(in_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

    def _make_heads(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        d = self.hidden_dim
        return _GradeHead(d), _DropoutHead(d), _EngagementHead(d)

    # ---- Feature helpers --------------------------------------------------

    @staticmethod
    def _safe_text(node_store, n: int, text_dim: int) -> Tensor:
        """Return text_x if available, otherwise zeros."""
        tx = getattr(node_store, "text_x", None)
        if tx is not None:
            return tx
        return torch.zeros(n, text_dim)

    @staticmethod
    def _safe_behavior(node_store) -> Optional[Tensor]:
        b = getattr(node_store, "behavior", None)
        return b if (b is not None and b.numel() > 0) else None


# ---------------------------------------------------------------------------
# 2. GATBaseline
# ---------------------------------------------------------------------------


class GATBaseline(_NeuralBaseline):
    """2-layer GAT on a homogeneous student-student co-enrollment graph.

    All students enrolled in the same course are connected.  The adjacency is
    computed once during :meth:`fit` via sparse matrix multiplication and
    cached for subsequent :meth:`predict` calls.

    Student features (structured + text + behavioral) are projected to
    ``hidden_dim`` before being passed to the GAT convolutions.

    Args:
        hidden_dim: Node embedding dimension.
        num_heads: Attention heads per GATConv layer.
        dropout: Dropout on attention weights and embeddings.
        **kwargs: Forwarded to :class:`_NeuralBaseline`.
    """

    def _build(self, data: HeteroData) -> None:
        s = data["student"]
        d = self.hidden_dim

        struct_dim = s.x.size(-1)
        text_dim   = s.text_x.size(-1) if hasattr(s, "text_x") and s.text_x is not None else 768
        behav_dim  = s.behavior.size(-1) if hasattr(s, "behavior") and s.behavior is not None else 2

        # Per-modality encoders
        self.struct_enc = StructuredEncoder(struct_dim, d * 2, d)
        self.text_proj_ = self._text_proj(text_dim)
        self.behav_enc  = BehavioralEncoder(behav_dim, d // 2, d, num_layers=2)
        # Fuse three d-dim modalities → d
        self.student_fuse = self._fuse_block(d * 3)

        # GATConv: d → (d // num_heads) × num_heads = d
        head_dim = max(1, d // self.num_heads)
        self.convs = nn.ModuleList([
            GATConv(
                in_channels=d,
                out_channels=head_dim,
                heads=self.num_heads,
                concat=True,
                dropout=self.dropout_rate,
            )
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])

        self.grade_head, self.dropout_head, self.engagement_head = self._make_heads()

        # Precompute co-enrollment graph (CPU tensor, moved to device in forward)
        self._coenroll_ei: Tensor = self._build_coenrollment(data)
        self._text_dim = text_dim

    @staticmethod
    def _build_coenrollment(data: HeteroData) -> Tensor:
        """Student-student edges: students sharing at least one enrolled course."""
        store = data["student", "enrolled_in", "course"]
        if not (hasattr(store, "edge_index") and store.edge_index is not None
                and store.edge_index.numel() > 0):
            return torch.zeros(2, 0, dtype=torch.long)

        ei  = store.edge_index.cpu()          # (2, E): row0=student, row1=course
        n_s = data["student"].x.size(0)
        n_c = data["course"].x.size(0)

        if _HAS_SCIPY:
            src = ei[0].numpy()
            dst = ei[1].numpy()
            A   = _sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_s, n_c))
            SCS = (A @ A.T).tocoo()
            mask = SCS.row != SCS.col                     # remove self-loops
            row  = torch.tensor(SCS.row[mask], dtype=torch.long)
            col  = torch.tensor(SCS.col[mask], dtype=torch.long)
            return torch.stack([row, col])
        else:
            # Fallback: re-use direct collaborated_with edges if available
            collab = data.get(("student", "collaborated_with", "student"), None)
            if collab is not None and hasattr(collab, "edge_index"):
                return collab.edge_index.cpu()
            return torch.zeros(2, 0, dtype=torch.long)

    def _encode_students(self, data: HeteroData) -> Tensor:
        s   = data["student"]
        dev = s.x.device

        h_s = self.struct_enc(s.x)
        h_t = self.text_proj_(self._safe_text(s, s.x.size(0), self._text_dim).to(dev))

        behavior = self._safe_behavior(s)
        h_b = self.behav_enc(behavior) if behavior is not None else h_t.detach()

        return self.student_fuse(torch.cat([h_s, h_t, h_b], dim=-1))

    def _forward_impl(self, data: HeteroData) -> Dict[str, Tensor]:
        x          = self._encode_students(data)
        edge_index = self._coenroll_ei.to(x.device)

        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(x + conv(x, edge_index)))

        return {
            "grade":      self.grade_head(x),
            "dropout":    self.dropout_head(x),
            "engagement": self.engagement_head(x),
        }


# ---------------------------------------------------------------------------
# 3. HGTBaseline
# ---------------------------------------------------------------------------


class HGTBaseline(_NeuralBaseline):
    """Standard HGT on the full heterogeneous graph — no modality gate.

    Identical backbone to MG-HGNN (same HGTConv layers, same encoders, same
    prediction heads) but replaces the per-relation ModalityGate with a
    simple learnable linear fusion of the three student modalities:

        h_student = Linear([h_s ∥ h_t ∥ h_b]) + LayerNorm + ReLU

    This is the direct ablation baseline: it isolates the contribution of the
    modality gate by holding everything else constant.

    Args:
        hidden_dim: Shared embedding dimension.
        num_heads: HGTConv attention heads.  Must divide ``hidden_dim``.
        dropout: Dropout rate.
        **kwargs: Forwarded to :class:`_NeuralBaseline`.
    """

    def _build(self, data: HeteroData) -> None:
        d = self.hidden_dim

        s = data["student"]
        struct_dim    = s.x.size(-1)
        text_dim      = s.text_x.size(-1) if hasattr(s, "text_x") and s.text_x is not None else 768
        behav_dim     = s.behavior.size(-1) if hasattr(s, "behavior") and s.behavior is not None else 2
        inst_struct   = data["instructor"].x.size(-1) if "instructor" in data.node_types else struct_dim

        # Student encoders (one per modality — mirrors MGHGNN, minus the gate)
        self.struct_enc   = StructuredEncoder(struct_dim,  d * 2, d)
        self.text_proj_   = self._text_proj(text_dim)
        self.behav_enc    = BehavioralEncoder(behav_dim, d // 2, d, num_layers=2)
        # Fusion: cat(h_s, h_t, h_b) ∈ ℝ^{3d} → ℝ^d
        self.student_fuse = self._fuse_block(d * 3)

        # Non-student node encoders
        self.course_enc    = self._text_proj(text_dim)
        self.resource_enc  = self._text_proj(text_dim)
        self.instructor_enc = StructuredEncoder(inst_struct, d, d)

        # HGTConv layers + per-type LayerNorms (same as MGHGNN)
        self.convs = nn.ModuleList([
            HGTConv(d, d, _HGNN_METADATA, heads=self.num_heads)
            for _ in range(2)
        ])
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(d) for nt in _NODE_TYPES})
            for _ in range(2)
        ])

        self.grade_head, self.dropout_head, self.engagement_head = self._make_heads()
        self._text_dim = text_dim

    def _encode_all_nodes(self, data: HeteroData) -> Dict[str, Tensor]:
        s   = data["student"]
        dev = s.x.device
        n_s = s.x.size(0)

        h_s = self.struct_enc(s.x)
        h_t = self.text_proj_(self._safe_text(s, n_s, self._text_dim).to(dev))

        behavior = self._safe_behavior(s)
        h_b = self.behav_enc(behavior) if behavior is not None else h_t.detach()

        h_student = self.student_fuse(torch.cat([h_s, h_t, h_b], dim=-1))

        n_c = data["course"].x.size(0)
        n_r = data["resource"].x.size(0)
        n_i = data["instructor"].x.size(0) if "instructor" in data.node_types else 0

        h_course    = self.course_enc(self._safe_text(data["course"],    n_c, self._text_dim).to(dev))
        h_resource  = self.resource_enc(self._safe_text(data["resource"], n_r, self._text_dim).to(dev))
        h_instructor = (
            self.instructor_enc(data["instructor"].x)
            if n_i > 0
            else torch.zeros(0, self.hidden_dim, device=dev)
        )

        return {
            "student":    h_student,
            "course":     h_course,
            "resource":   h_resource,
            "instructor": h_instructor,
        }

    def _build_edge_index_dict(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str, str], Tensor]:
        result: Dict = {}
        for et in _EDGE_TYPES:
            store = data[et]
            if (hasattr(store, "edge_index")
                    and store.edge_index is not None
                    and store.edge_index.numel() > 0):
                result[et] = store.edge_index
        return result

    def _forward_impl(self, data: HeteroData) -> Dict[str, Tensor]:
        x_dict         = self._encode_all_nodes(data)
        edge_index_dict = self._build_edge_index_dict(data)

        for conv, norm_dict in zip(self.convs, self.layer_norms):
            x_new = conv(x_dict, edge_index_dict)
            for nt in _NODE_TYPES:
                h_new = x_new.get(nt)
                if h_new is not None:
                    x_dict[nt] = F.relu(norm_dict[nt](x_dict[nt] + h_new))

        student_emb = x_dict["student"]
        return {
            "grade":      self.grade_head(student_emb),
            "dropout":    self.dropout_head(student_emb),
            "engagement": self.engagement_head(student_emb),
        }


# ---------------------------------------------------------------------------
# 4. HANBaseline
# ---------------------------------------------------------------------------


class HANBaseline(_NeuralBaseline):
    """HAN with two meta-paths: student-course-student (SCS) and
    student-resource-student (SRS).

    Two HANConv layers aggregate over the SCS and SRS meta-paths with
    node-level and semantic-level attention.  The reverse edges
    ``("course", "rev_enrolled_in", "student")`` and
    ``("resource", "rev_accessed", "student")`` are added dynamically
    inside :meth:`_forward_impl` from the original edge_index tensors.

    Requires ``torch_geometric >= 2.0`` for :class:`~torch_geometric.nn.HANConv`.

    Args:
        hidden_dim: Node embedding dimension.
        num_heads: Attention heads in each HANConv layer.
        dropout: Attention dropout.
        **kwargs: Forwarded to :class:`_NeuralBaseline`.

    Raises:
        ImportError: If ``torch_geometric.nn.HANConv`` is not available.
    """

    def __init__(self, **kwargs) -> None:
        if not _HAS_HANCONV:
            raise ImportError(
                "HANBaseline requires torch_geometric >= 2.0 with HANConv. "
                "Upgrade with: pip install torch-geometric --upgrade"
            )
        super().__init__(**kwargs)

    def _build(self, data: HeteroData) -> None:
        d = self.hidden_dim

        s = data["student"]
        struct_dim = s.x.size(-1)
        text_dim   = s.text_x.size(-1) if hasattr(s, "text_x") and s.text_x is not None else 768
        behav_dim  = s.behavior.size(-1) if hasattr(s, "behavior") and s.behavior is not None else 2

        # Student encoders (same as HGTBaseline)
        self.struct_enc   = StructuredEncoder(struct_dim, d * 2, d)
        self.text_proj_   = self._text_proj(text_dim)
        self.behav_enc    = BehavioralEncoder(behav_dim, d // 2, d, num_layers=2)
        self.student_fuse = self._fuse_block(d * 3)

        # Course and resource encoders
        self.course_enc   = self._text_proj(text_dim)
        self.resource_enc = self._text_proj(text_dim)

        # HANConv layers — only the three node types participating in SCS/SRS
        in_channels = {nt: d for nt in _HAN_NODE_TYPES}
        self.han_convs = nn.ModuleList([
            _HANConv(
                in_channels=in_channels,
                out_channels=d,
                metadata=_HAN_METADATA,
                heads=self.num_heads,
                dropout=self.dropout_rate,
            )
            for _ in range(2)
        ])
        self.han_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])

        self.grade_head, self.dropout_head, self.engagement_head = self._make_heads()
        self._text_dim = text_dim

    def _encode_nodes(self, data: HeteroData) -> Dict[str, Tensor]:
        dev = data["student"].x.device
        s   = data["student"]
        n_s = s.x.size(0)
        n_c = data["course"].x.size(0)
        n_r = data["resource"].x.size(0)

        h_s = self.struct_enc(s.x)
        h_t = self.text_proj_(self._safe_text(s, n_s, self._text_dim).to(dev))

        behavior = self._safe_behavior(s)
        h_b = self.behav_enc(behavior) if behavior is not None else h_t.detach()

        return {
            "student":  self.student_fuse(torch.cat([h_s, h_t, h_b], dim=-1)),
            "course":   self.course_enc(self._safe_text(data["course"],   n_c, self._text_dim).to(dev)),
            "resource": self.resource_enc(self._safe_text(data["resource"], n_r, self._text_dim).to(dev)),
        }

    def _build_han_edge_index_dict(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str, str], Tensor]:
        """Build forward + reverse edge index dict for both meta-paths."""
        result: Dict = {}

        def _add(fwd_key: Tuple, rev_rel: str) -> None:
            src_t, rel, dst_t = fwd_key
            store = data[fwd_key]
            if not (hasattr(store, "edge_index")
                    and store.edge_index is not None
                    and store.edge_index.numel() > 0):
                return
            ei = store.edge_index
            result[fwd_key]                     = ei
            result[(dst_t, rev_rel, src_t)]     = ei.flip(0)   # reverse

        _add(("student", "enrolled_in", "course"),  "rev_enrolled_in")
        _add(("student", "accessed",    "resource"), "rev_accessed")

        return result

    def _forward_impl(self, data: HeteroData) -> Dict[str, Tensor]:
        x_dict         = self._encode_nodes(data)
        edge_index_dict = self._build_han_edge_index_dict(data)

        for han_conv, norm in zip(self.han_convs, self.han_norms):
            x_new = han_conv(x_dict, edge_index_dict)
            h_new = x_new.get("student")
            if h_new is not None:
                x_dict["student"] = F.relu(norm(x_dict["student"] + h_new))

        student_emb = x_dict["student"]
        return {
            "grade":      self.grade_head(student_emb),
            "dropout":    self.dropout_head(student_emb),
            "engagement": self.engagement_head(student_emb),
        }
