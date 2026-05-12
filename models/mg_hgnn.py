"""Full MG-HGNN model with PyTorch Geometric HGTConv backbone.

Architecture
------------
1. **Modality encoders** — per node type:
   * Student → StructuredEncoder (tabular x), TextEncoder projection
     (pre-computed text_x), BehavioralEncoder (click-sequence behavior).
   * Course / Resource → lightweight text projection on pre-computed text_x.
   * Instructor → StructuredEncoder (tabular x).

2. **Pre-gating** — for each of the four student-sourced edge types, the
   shared ModalityGate computes a relation-specific weighted combination of
   the student's three modality embeddings.  The four gated vectors are
   mean-pooled into a single initial student embedding.  This per-relation
   signal is what distinguishes MG-HGNN from HGT.

3. **Two HGTConv layers** — the pre-gated x_dict is passed through two
   HGTConv layers with residual connections and per-node-type LayerNorm.

4. **Three prediction heads** on the final student embedding:
   * ``GradeHead``       — Linear(d, 1)               (regression)
   * ``DropoutRiskHead`` — Linear(d, 1) + Sigmoid      (binary probability)
   * ``EngagementHead``  — Linear(d, 3)                (3-class logits)

``forward()`` returns a ``dict`` with keys ``"grade"``, ``"dropout"``,
``"engagement"``, allowing all three losses to be computed jointly.

Note on TextEncoder
-------------------
``TextEncoder`` (which wraps BERT) is instantiated once for students.  Its
``.proj`` sub-module is called directly when pre-computed 768-d embeddings
are stored in ``data["student"].text_x`` (the default from the data loaders).
To switch to online BERT inference, call
``model.student_text_enc(input_ids, attention_mask)`` instead.  Courses and
resources use a lightweight projection that mirrors the TextEncoder ``.proj``
structure without loading a second BERT instance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from .encoders import BehavioralEncoder, StructuredEncoder, TextEncoder
from .gate import ModalityGate


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------


class GradeHead(nn.Module):
    """Regression head for grade prediction.

    Args:
        in_dim: Input embedding dimension *d*.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Args:  x ``(N, d)``  →  Returns ``(N,)`` raw regression scores."""
        return self.linear(x).squeeze(-1)


class DropoutRiskHead(nn.Module):
    """Binary dropout-risk classification head.

    Applies ``Linear(d, 1)`` followed by ``Sigmoid`` to produce a dropout
    probability in ``[0, 1]``.  Use ``BCELoss`` (not ``BCEWithLogitsLoss``)
    for training since Sigmoid is applied inside the head.

    Args:
        in_dim: Input embedding dimension *d*.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x ``(N, d)``  →  Returns ``(N,)`` dropout probabilities."""
        return self.net(x).squeeze(-1)


class EngagementHead(nn.Module):
    """Three-class engagement-level classification head.

    Outputs raw logits suitable for ``CrossEntropyLoss``.

    Args:
        in_dim: Input embedding dimension *d*.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x ``(N, d)``  →  Returns ``(N, 3)`` class logits."""
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MGHGNN(nn.Module):
    """Modality-Gated Heterogeneous Graph Neural Network.

    Follows the four-stage pipeline described in the paper:
    encode → pre-gate → HGTConv × 2 → predict.

    Args:
        student_struct_dim: Feature width of ``data["student"].x``.
        instructor_struct_dim: Feature width of ``data["instructor"].x``.
        hidden_dim: Shared embedding dimension *d* used throughout.
            Must be divisible by ``num_heads``.
        num_heads: Number of attention heads in each HGTConv layer.
        num_conv_layers: Number of HGTConv message-passing layers.
            Default ``2`` as specified in the paper.
        dropout: Dropout rate applied in encoders and prediction heads.
        behavioral_input_dim: Feature width of each time-step in the
            student click-sequence (2 for ``(resource_idx, log_clicks)``).
        text_embed_dim: Dimensionality of pre-computed BERT embeddings
            stored in ``data[*].text_x`` (768 for BERT-base).
        freeze_bert: If ``True`` (default), BERT parameters are frozen
            and only the projection head is trained.
    """

    NODE_TYPES: List[str] = ["student", "course", "instructor", "resource"]
    EDGE_TYPES: List[Tuple[str, str, str]] = [
        ("student", "enrolled_in",       "course"),
        ("student", "collaborated_with", "student"),
        ("student", "accessed",          "resource"),
        ("student", "submitted_to",      "course"),
    ]
    # All four edge types have students as source → gate always operates on students.
    STUDENT_RELATIONS: List[str] = [
        "enrolled_in", "collaborated_with", "accessed", "submitted_to"
    ]
    METADATA: Tuple = (NODE_TYPES, EDGE_TYPES)

    def __init__(
        self,
        student_struct_dim: int,
        instructor_struct_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_conv_layers: int = 2,
        dropout: float = 0.3,
        behavioral_input_dim: int = 2,
        text_embed_dim: int = 768,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
        )
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers

        # ----------------------------------------------------------------
        # Encoders
        # ----------------------------------------------------------------

        # Students: all three modalities.
        # StructuredEncoder: tabular x → h_s
        self.student_struct_enc = StructuredEncoder(
            input_dim=student_struct_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
        )
        # TextEncoder: contains a BERT backbone + proj head.
        # In forward() we call enc.proj(text_x) on pre-computed BERT embeddings
        # to avoid a redundant BERT forward pass.  Swap to
        # enc(input_ids, attention_mask) when tokenized input is available.
        self.student_text_enc = TextEncoder(
            output_dim=hidden_dim,
            freeze_bert=freeze_bert,
        )
        # BehavioralEncoder: click-sequence → h_b
        self.student_behav_enc = BehavioralEncoder(
            input_dim=behavioral_input_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim,
            num_layers=2,
        )

        # Courses: TextEncoder-style projection on pre-computed text_x.
        # A full BERT instance is NOT loaded here to avoid redundancy; the
        # student TextEncoder's projection architecture is mirrored.
        self.course_text_enc = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Resources: same as courses.
        self.resource_text_enc = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Instructors: structured features only.
        self.instructor_struct_enc = StructuredEncoder(
            input_dim=instructor_struct_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        # ----------------------------------------------------------------
        # ModalityGate — one gate bank for all four student-sourced relations.
        # ----------------------------------------------------------------
        self.gate = ModalityGate(
            embed_dim=hidden_dim,
            edge_types=self.STUDENT_RELATIONS,
        )

        # ----------------------------------------------------------------
        # HGTConv layers + per-node-type LayerNorms.
        # ----------------------------------------------------------------
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=self.METADATA,
                heads=num_heads,
            )
            for _ in range(num_conv_layers)
        ])
        # One LayerNorm per node type per layer for stable post-conv residuals.
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.NODE_TYPES})
            for _ in range(num_conv_layers)
        ])

        # ----------------------------------------------------------------
        # Prediction heads — all three are always computed.
        # ----------------------------------------------------------------
        self.grade_head      = GradeHead(hidden_dim)
        self.dropout_head    = DropoutRiskHead(hidden_dim)
        self.engagement_head = EngagementHead(hidden_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        """Run a full forward pass from raw features to all three predictions.

        Args:
            data: :class:`~torch_geometric.data.HeteroData` with:

            * ``data["student"].x``         — ``(N_s, student_struct_dim)``
            * ``data["student"].text_x``    — ``(N_s, 768)`` pre-computed BERT
            * ``data["student"].behavior``  — ``(N_s, seq_len, 2)`` optional
            * ``data["course"].text_x``     — ``(N_c, 768)``
            * ``data["resource"].text_x``   — ``(N_r, 768)``
            * ``data["instructor"].x``      — ``(N_i, instructor_struct_dim)``
            * ``data[edge_type].edge_index`` for each of the four edge types.

        Returns:
            Dictionary with three entries:

            * ``"grade"``      — ``(N_s,)``   raw regression scores.
            * ``"dropout"``    — ``(N_s,)``   dropout probabilities in ``[0, 1]``.
            * ``"engagement"`` — ``(N_s, 3)`` class logits.
        """
        # ---- Step 1: encode all node types ------------------------------
        h_s, h_t, h_b, other_embs = self._encode_nodes(data)

        # ---- Step 2: pre-gate student embeddings per relation type ------
        h_student_init = self._pre_gate(h_s, h_t, h_b)

        # ---- Step 3: initial embedding dict for HGTConv -----------------
        x_dict: Dict[str, Tensor] = {
            "student":    h_student_init,
            "course":     other_embs["course"],
            "resource":   other_embs["resource"],
            "instructor": other_embs["instructor"],
        }

        # ---- Step 4: edge index dict (skip empty edge types) ------------
        edge_index_dict = self._build_edge_index_dict(data)

        # ---- Step 5: HGTConv layers with residual + LayerNorm -----------
        for conv, norm_dict in zip(self.convs, self.norms):
            x_new: Dict[str, Tensor] = conv(x_dict, edge_index_dict)
            # Residual add + norm; node types with no incoming edges keep
            # their previous embedding unchanged (e.g. instructor).
            for nt in self.NODE_TYPES:
                h_new: Optional[Tensor] = x_new.get(nt)
                if h_new is not None:
                    x_dict[nt] = F.relu(norm_dict[nt](x_dict[nt] + h_new))

        # ---- Step 6: prediction heads on final student embeddings -------
        student_emb: Tensor = x_dict["student"]   # (N_s, d)
        return {
            "grade":      self.grade_head(student_emb),
            "dropout":    self.dropout_head(student_emb),
            "engagement": self.engagement_head(student_emb),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_nodes(
        self,
        data: HeteroData,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Encode every node type and return student modality tensors separately.

        Returns:
            h_s: Structured student embedding ``(N_s, d)``.
            h_t: Textual student embedding ``(N_s, d)``.
            h_b: Behavioral student embedding ``(N_s, d)``.
            other: Dict of initial embeddings for non-student node types.
        """
        # Students — three modalities
        h_s: Tensor = self.student_struct_enc(data["student"].x)

        # Use the pre-computed text embedding directly through the projection
        # sub-module.  Call self.student_text_enc(input_ids, attn_mask) here
        # when tokenized inputs are available instead.
        h_t: Tensor = self.student_text_enc.proj(data["student"].text_x)

        behavior: Optional[Tensor] = getattr(data["student"], "behavior", None)
        if behavior is not None and behavior.numel() > 0:
            h_b: Tensor = self.student_behav_enc(behavior)
        else:
            # Fall back to a copy of the text embedding when no sequence data.
            h_b = h_t.detach().clone()

        # Non-student nodes — single modality each
        other: Dict[str, Tensor] = {
            "course":     self.course_text_enc(data["course"].text_x),
            "resource":   self.resource_text_enc(data["resource"].text_x),
            "instructor": self.instructor_struct_enc(data["instructor"].x),
        }
        return h_s, h_t, h_b, other

    def _pre_gate(
        self,
        h_s: Tensor,
        h_t: Tensor,
        h_b: Tensor,
    ) -> Tensor:
        """Apply ModalityGate per relation type and mean-pool into one embedding.

        For each of the four student-sourced relation types, the shared
        ModalityGate computes a relation-specific weighted combination
        ``α_s·h_s + α_t·h_t + α_b·h_b`` for *every student*.  The four
        resulting tensors are stacked and averaged, producing the initial
        student embedding that is passed as node features to HGTConv.

        Args:
            h_s: ``(N_s, d)`` structured student embeddings.
            h_t: ``(N_s, d)`` textual student embeddings.
            h_b: ``(N_s, d)`` behavioral student embeddings.

        Returns:
            ``(N_s, d)`` mean-pooled gated embedding.
        """
        gated_per_rel: List[Tensor] = [
            self.gate(rel, h_s, h_t, h_b) for rel in self.STUDENT_RELATIONS
        ]
        # Stack → (num_relations, N_s, d), then mean over relation dim.
        return torch.stack(gated_per_rel, dim=0).mean(dim=0)

    def _build_edge_index_dict(
        self,
        data: HeteroData,
    ) -> Dict[Tuple[str, str, str], Tensor]:
        """Build the ``edge_index_dict`` required by HGTConv.

        Silently skips edge types that are missing from the graph or have
        an empty edge tensor (e.g. a batch with no collaborated_with links).

        Returns:
            Mapping from edge-type triple to ``LongTensor[2, E]``.
        """
        result: Dict[Tuple[str, str, str], Tensor] = {}
        for et in self.EDGE_TYPES:
            store = data[et]
            if hasattr(store, "edge_index") and store.edge_index is not None:
                if store.edge_index.numel() > 0:
                    result[et] = store.edge_index
        return result

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inspect_gate_weights(
        self, data: HeteroData
    ) -> Dict[str, List[float]]:
        """Return the mean gate weights per relation type for a data batch.

        Runs ``_encode_nodes`` then queries ``ModalityGate.get_gate_weights``
        in data mode (mean over the batch).  Useful for the qualitative
        analysis in Section 5 of the paper.

        Args:
            data: A populated HeteroData graph.

        Returns:
            ``{relation_name: [α_s, α_t, α_b]}`` as Python float lists.
        """
        self.eval()
        h_s, h_t, h_b, _ = self._encode_nodes(data)
        return {
            rel: self.gate.get_gate_weights(rel, h_s, h_t, h_b).tolist()
            for rel in self.STUDENT_RELATIONS
        }

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print and return a formatted parameter-count table per component.

        Distinguishes trainable vs. frozen parameters (BERT weights are frozen
        when ``freeze_bert=True``).

        Returns:
            The formatted summary string (also printed to stdout).
        """

        def _count(module: nn.Module) -> Tuple[int, int]:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen    = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            return trainable, frozen

        groups: Dict[str, nn.Module] = {
            # Encoders
            "StructuredEncoder   (student)":  self.student_struct_enc,
            "TextEncoder         (student)":  self.student_text_enc,
            "BehavioralEncoder   (student)":  self.student_behav_enc,
            "TextEncoder proj    (course)":   self.course_text_enc,
            "TextEncoder proj    (resource)": self.resource_text_enc,
            "StructuredEncoder   (instructor)":self.instructor_struct_enc,
            # Gate
            "ModalityGate":                   self.gate,
            # Graph convolutions
            **{
                f"HGTConv           [{i}]": conv
                for i, conv in enumerate(self.convs)
            },
            **{
                f"LayerNorms         [{i}]": norm
                for i, norm in enumerate(self.norms)
            },
            # Prediction heads
            "GradeHead":                      self.grade_head,
            "DropoutRiskHead":                self.dropout_head,
            "EngagementHead":                 self.engagement_head,
        }

        col_w = 42
        num_w = 12
        sep = "=" * (col_w + num_w + 4)

        lines: List[str] = [
            sep,
            f"{'MG-HGNN  —  Model Summary':^{col_w + num_w + 4}}",
            sep,
            f"  {'Component':<{col_w}} {'Parameters':>{num_w}}",
            "-" * (col_w + num_w + 4),
        ]

        total_train = total_frozen = 0
        for name, mod in groups.items():
            tr, fr = _count(mod)
            total_train  += tr
            total_frozen += fr
            note = f"  [{fr:,} frozen]" if fr else ""
            lines.append(f"  {name:<{col_w}} {tr + fr:>{num_w},}{note}")

        total = total_train + total_frozen
        lines += [
            "-" * (col_w + num_w + 4),
            f"  {'Trainable parameters':<{col_w}} {total_train:>{num_w},}",
            f"  {'Frozen parameters':<{col_w}} {total_frozen:>{num_w},}",
            f"  {'Total parameters':<{col_w}} {total:>{num_w},}",
            sep,
        ]

        result = "\n".join(lines)
        print(result)
        return result
