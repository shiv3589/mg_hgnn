import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from typing import Dict

from config import Config
from models.encoders import StructuredEncoder, TextEncoder, BehavioralEncoder
from models.gate import ModalityGate

# Full (src, rel, dst) triples derived from config edge_type semantics.
# All four relations originate from student nodes; collaborated_with is
# student→student (homogeneous), the rest are student→other.
_EDGE_TRIPLES = [
    ("student", "enrolled_in",       "course"),
    ("student", "collaborated_with", "student"),
    ("student", "accessed",          "resource"),
    ("student", "submitted_to",      "course"),
]


class MG_HGNN(nn.Module):
    """Modality-Gated Heterogeneous Graph Neural Network.

    Students carry three modality embeddings (structured, text, behavioral).
    Before graph convolution, a per-relation softmax gate fuses these three
    streams into a single node embedding conditioned on each edge relation.
    Course and resource nodes use text embeddings only; instructor nodes use
    a lightweight structured encoder.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        d = config.embed_dim

        # ---- Modality encoders ----
        self.struct_enc = StructuredEncoder(config)
        self.text_enc   = TextEncoder(config, freeze_bert=True)
        self.behav_enc  = BehavioralEncoder(config)

        # Instructor nodes: simple MLP (needed by HGTConv; not in spec)
        self.inst_enc = nn.Sequential(
            nn.Linear(config.structured_input_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # ---- Novel component: per-relation modality gate ----
        self.gate = ModalityGate(edge_types=config.edge_types, embed_dim=d)

        # ---- HGTConv layers ----
        metadata = (config.node_types, _EDGE_TRIPLES)
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=d,
                out_channels=d,
                metadata=metadata,
                heads=config.num_heads,
            )
            for _ in range(config.num_hgnn_layers)
        ])
        self.conv_drop = nn.Dropout(config.dropout)

        # ---- Prediction heads (applied to student nodes only) ----
        self.grade_head      = nn.Linear(d, 1)
        self.dropout_head    = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.engagement_head = nn.Linear(d, 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_students(self, data: HeteroData,
                         bert_cache: dict = None) -> torch.Tensor:
        """Fuse three modality embeddings via gating, averaged across edge types."""
        h_s = self.struct_enc(data["student"].x_struct)
        if bert_cache is not None and "student" in bert_cache:
            h_t = bert_cache["student"]
        else:
            h_t = self.text_enc(data["student"].input_ids, data["student"].attention_mask)
        h_b = self.behav_enc(data["student"].x_behav)

        # Gate produces a distinct fused vector per relation; average for HGTConv input.
        gated = [self.gate(r, h_s, h_t, h_b)[0] for r in self.cfg.edge_types]
        return torch.stack(gated, dim=0).mean(dim=0)   # (N_students, embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: HeteroData,
                bert_cache: dict = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            data       : PyG HeteroData with student/course/resource/instructor nodes
            bert_cache : optional dict {node_type: FloatTensor (N, embed_dim)}
                         pre-computed by cache_embeddings.py.  When provided,
                         TextEncoder is bypassed for cached node types — speeds
                         up training ~10× on datasets with frozen BERT.
        Returns:
            {'grade': (N,1), 'dropout': (N,1), 'engagement': (N,3)}
        """
        def _text(node_type: str) -> torch.Tensor:
            if bert_cache is not None and node_type in bert_cache:
                return bert_cache[node_type]
            return self.text_enc(
                data[node_type].input_ids,
                data[node_type].attention_mask,
            )

        x_dict: Dict[str, torch.Tensor] = {
            "student":    self._encode_students(data, bert_cache),
            "course":     _text("course"),
            "resource":   _text("resource"),
            "instructor": self.inst_enc(data["instructor"].x_struct),
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.conv_drop(F.relu(v)) for k, v in x_dict.items()}

        s = x_dict["student"]
        return {
            "grade":      self.grade_head(s),        # (N, 1)  regression
            "dropout":    self.dropout_head(s),      # (N, 1)  binary probability
            "engagement": self.engagement_head(s),   # (N, 3)  3-class logits
        }

    # ------------------------------------------------------------------
    # Mini-batch / full-graph encode (no prediction heads)
    # ------------------------------------------------------------------

    def _bert_lookup(self, node_type: str, data: HeteroData,
                     bert_cache: dict) -> torch.Tensor:
        """Return text embeddings; handles full-graph and NeighborLoader mini-batches."""
        if bert_cache is not None and node_type in bert_cache:
            store = data[node_type]
            if hasattr(store, "n_id"):
                return bert_cache[node_type][store.n_id]
            return bert_cache[node_type]
        return self.text_enc(data[node_type].input_ids,
                             data[node_type].attention_mask)

    def encode_all(self, data: HeteroData,
                   bert_cache: dict = None) -> Dict[str, torch.Tensor]:
        """Run encoders + gate + HGTConv; return node-embedding dict (no heads).

        Works for both full-graph HeteroData and NeighborLoader mini-batches.
        In mini-batch mode, bert_cache entries are indexed by n_id automatically.
        Caller is responsible for context (no_grad or grad as needed).
        """
        h_s = self.struct_enc(data["student"].x_struct)
        h_t = self._bert_lookup("student", data, bert_cache)
        h_b = self.behav_enc(data["student"].x_behav)
        gated = [self.gate(r, h_s, h_t, h_b)[0] for r in self.cfg.edge_types]

        x_dict: Dict[str, torch.Tensor] = {
            "student": torch.stack(gated, dim=0).mean(dim=0),
        }
        for ntype in ("course", "resource"):
            if ntype in data.node_types:
                x_dict[ntype] = self._bert_lookup(ntype, data, bert_cache)
        if "instructor" in data.node_types and hasattr(data["instructor"], "x_struct"):
            x_dict["instructor"] = self.inst_enc(data["instructor"].x_struct)

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.conv_drop(F.relu(v)) for k, v in x_dict.items()}

        return x_dict

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def param_summary(self) -> None:
        """Print trainable param count and share per component."""
        components = {
            "struct_enc":      self.struct_enc,
            "text_enc":        self.text_enc,
            "behav_enc":       self.behav_enc,
            "inst_enc":        self.inst_enc,
            "gate":            self.gate,
            "convs[0]":        self.convs[0],
            "convs[1]":        self.convs[1],
            "grade_head":      self.grade_head,
            "dropout_head":    self.dropout_head,
            "engagement_head": self.engagement_head,
        }

        total_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        W = 58
        print("\n" + "=" * W)
        print(f"  {'Component':<22} {'Trainable':>10}   {'% of Total':>10}")
        print("=" * W)
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            pct = 100.0 * n / total_train if total_train else 0.0
            print(f"  {name:<22} {n:>10,}   {pct:>9.2f}%")
        print("-" * W)
        print(f"  {'TOTAL (trainable)':<22} {total_train:>10,}   {'100.00%':>10}")
        print(f"  {'TOTAL (frozen/BERT)':<22} {total_frozen:>10,}")
        print(f"  {'TOTAL (all params)':<22} {total_train + total_frozen:>10,}")
        print("=" * W)
        gate_n   = sum(p.numel() for p in self.gate.parameters() if p.requires_grad)
        gate_pct = 100.0 * gate_n / total_train if total_train else 0.0
        verdict  = "OK (<1%)" if gate_pct < 1.0 else "WARNING (>=1%)"
        print(f"  Gate share: {gate_pct:.3f}%  [{verdict}]")
        print("=" * W + "\n")


# ----------------------------------------------------------------
# Ablation variants — gate behaviour only; all else identical
# ----------------------------------------------------------------

class MG_HGNN_UniformGate(MG_HGNN):
    """Ablation 1: gate frozen at α = [1/3, 1/3, 1/3] for all edge types."""

    def __init__(self, config: Config):
        super().__init__(config)
        for param in self.gate.parameters():
            param.requires_grad = False

        def _uniform(edge_type: str, h_s: torch.Tensor,
                     h_t: torch.Tensor, h_b: torch.Tensor):
            N = h_s.shape[0]
            alpha = torch.full((N, 3), 1 / 3, device=h_s.device)
            fused = alpha[:, 0:1] * h_s + alpha[:, 1:2] * h_t + alpha[:, 2:3] * h_b
            return fused, alpha

        self.gate.forward = _uniform


class MG_HGNN_SharedGate(MG_HGNN):
    """Ablation 2: single shared gate W (no per-relation conditioning)."""

    def __init__(self, config: Config):
        super().__init__(config)
        d = config.embed_dim
        self.shared_gate_w = nn.Linear(3 * d, 3)
        for param in self.gate.parameters():
            param.requires_grad = False

        shared_w = self.shared_gate_w   # closure reference

        def _shared(edge_type: str, h_s: torch.Tensor,
                    h_t: torch.Tensor, h_b: torch.Tensor):
            import torch.nn.functional as _F
            combined = torch.cat([h_s, h_t, h_b], dim=-1)
            alpha = _F.softmax(shared_w(combined), dim=-1)
            fused = alpha[:, 0:1] * h_s + alpha[:, 1:2] * h_t + alpha[:, 2:3] * h_b
            return fused, alpha

        self.gate.forward = _shared


class MG_HGNN_StructOnly(MG_HGNN):
    """Ablation 3: structured modality only (α = [1, 0, 0])."""

    def __init__(self, config: Config):
        super().__init__(config)
        for param in self.gate.parameters():
            param.requires_grad = False

        def _struct(edge_type: str, h_s: torch.Tensor,
                    h_t: torch.Tensor, h_b: torch.Tensor):
            N = h_s.shape[0]
            alpha = torch.zeros(N, 3, device=h_s.device)
            alpha[:, 0] = 1.0
            return h_s, alpha

        self.gate.forward = _struct


class MG_HGNN_BehavOnly(MG_HGNN):
    """Ablation 4: behavioral modality only (α = [0, 0, 1])."""

    def __init__(self, config: Config):
        super().__init__(config)
        for param in self.gate.parameters():
            param.requires_grad = False

        def _behav(edge_type: str, h_s: torch.Tensor,
                   h_t: torch.Tensor, h_b: torch.Tensor):
            N = h_s.shape[0]
            alpha = torch.zeros(N, 3, device=h_s.device)
            alpha[:, 2] = 1.0
            return h_b, alpha

        self.gate.forward = _behav
