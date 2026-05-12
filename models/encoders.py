"""Modality-specific node encoders for MG-HGNN.

Each encoder maps raw node features to a shared *d*-dimensional embedding
and corresponds to one row of the modality-encoder table in the paper:

* :class:`StructuredEncoder`  — h_s(v) via a 2-layer BatchNorm MLP.
* :class:`TextEncoder`        — h_t(v) via BERT + masked mean-pooling.
* :class:`BehavioralEncoder`  — h_b(v) via a 2-layer BiLSTM.

All three classes expose the same calling convention:
``encoder(...)  →  FloatTensor[batch_size, output_dim]``.

Compatibility note
------------------
:class:`TextEncoder` accepts tokenised inputs ``(input_ids, attention_mask)``
produced by a HuggingFace tokenizer, *not* pre-computed embedding vectors.
If you store raw text in node features, run the tokenizer offline and cache
the token tensors alongside the graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. StructuredEncoder
# ---------------------------------------------------------------------------


class StructuredEncoder(nn.Module):
    """Two-layer MLP with BatchNorm and ReLU for tabular node features.

    Architecture::

        Linear(input_dim → hidden_dim)
        BatchNorm1d(hidden_dim)
        ReLU
        Dropout(0.3)
        Linear(hidden_dim → output_dim)

    The ``Dropout(0.3)`` is placed *before* the output linear layer so that
    the final projection still acts on a stochastically regularised
    representation during training.

    .. note::
        ``BatchNorm1d`` requires ``batch_size ≥ 2`` in training mode.
        Use ``model.eval()`` when running single-sample inference.

    Args:
        input_dim: Dimensionality of the raw tabular feature vector.
        hidden_dim: Width of the intermediate hidden layer.
        output_dim: Output embedding dimension *d*. Default: ``128``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of tabular feature vectors.

        Args:
            x: Float tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Float tensor of shape ``(batch_size, output_dim)``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. TextEncoder
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """BERT-based text encoder with masked mean-pooling and linear projection.

    Loads ``bert-base-uncased`` from the HuggingFace Hub, passes tokenised
    input through the transformer, and produces a single sentence embedding
    by averaging the last hidden state over *non-padding* token positions.
    The 768-d pooled vector is then projected to ``output_dim`` via a linear
    layer followed by LayerNorm.

    Architecture::

        BertModel (bert-base-uncased, 12 layers, hidden=768)
          ↓  last_hidden_state  [B, seq_len, 768]
        masked mean-pool over seq_len  →  [B, 768]
        Linear(768 → output_dim)
        LayerNorm(output_dim)

    Args:
        output_dim: Target embedding dimension *d*. Default: ``128``.
        freeze_bert: If ``True`` (default), all BERT parameters are frozen
            and only the projection head is trained.  Set to ``False`` to
            fine-tune BERT end-to-end.
    """

    _BERT_MODEL = "bert-base-uncased"
    _BERT_HIDDEN = 768

    def __init__(
        self,
        output_dim: int = 128,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "TextEncoder requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from exc

        self.bert = AutoModel.from_pretrained(self._BERT_MODEL)

        if freeze_bert:
            self.freeze_bert()

        self.proj = nn.Sequential(
            nn.Linear(self._BERT_HIDDEN, output_dim),
            nn.LayerNorm(output_dim),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def freeze_bert(self) -> None:
        """Freeze all BERT parameters (gradients disabled)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self) -> None:
        """Unfreeze all BERT parameters to allow end-to-end fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a batch of tokenised text sequences.

        Args:
            input_ids: Long tensor of shape ``(batch_size, seq_len)``
                containing WordPiece token IDs.
            attention_mask: Long tensor of shape ``(batch_size, seq_len)``
                with ``1`` for real tokens and ``0`` for padding.

        Returns:
            Float tensor of shape ``(batch_size, output_dim)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state          # (B, seq_len, 768)

        # Masked mean-pool: exclude padding positions from the average.
        mask = attention_mask.unsqueeze(-1).float()      # (B, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)         # (B, 768)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        pooled = summed / token_counts                   # (B, 768)

        return self.proj(pooled)                         # (B, output_dim)


# ---------------------------------------------------------------------------
# 3. BehavioralEncoder
# ---------------------------------------------------------------------------


class BehavioralEncoder(nn.Module):
    """Two-layer bidirectional LSTM for student activity sequences.

    Takes a zero-padded sequence of per-step feature vectors, runs them
    through a 2-layer BiLSTM, concatenates the final forward and backward
    hidden states from the *last* layer, applies ``Dropout(0.3)`` to that
    concatenation, and projects the result to ``output_dim``.

    Architecture::

        BiLSTM(input_dim, hidden_dim, num_layers=2, dropout=0.3)
          ↓  final hidden states: h_fwd [B, hidden_dim],  h_bwd [B, hidden_dim]
        concat  →  [B, hidden_dim * 2]
        Dropout(0.3)
        Linear(hidden_dim * 2 → output_dim)

    The ``dropout=0.3`` passed to :class:`nn.LSTM` is applied *between*
    layers (standard PyTorch behaviour); the explicit ``Dropout(0.3)`` after
    concatenation regularises the final representation going into the
    projection.

    Args:
        input_dim: Feature width at each time-step (e.g. ``2`` for
            ``(resource_idx, log_clicks)``).
        hidden_dim: Hidden state size *per direction* inside the LSTM.
            Default: ``64``.
        output_dim: Output embedding dimension *d*. Default: ``128``.
        num_layers: Number of stacked LSTM layers. Default: ``2``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,           # applied between LSTM layers
        )
        self.drop = nn.Dropout(p=0.3)  # applied to the concatenated final hidden state
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of activity sequences.

        Args:
            x: Float tensor of shape ``(batch_size, seq_len, input_dim)``.

        Returns:
            Float tensor of shape ``(batch_size, output_dim)``.
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch_size, hidden_dim)
        # Rows -2 and -1 are the last-layer forward and backward final states.
        h_fwd = h_n[-2]                              # (B, hidden_dim)
        h_bwd = h_n[-1]                              # (B, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, hidden_dim * 2)
        return self.proj(self.drop(h_cat))           # (B, output_dim)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    BATCH, SEQ_LEN, OUTPUT_DIM = 8, 32, 128

    # ------------------------------------------------------------------ 1
    print("Testing StructuredEncoder ...", end=" ", flush=True)
    enc_s = StructuredEncoder(input_dim=16, hidden_dim=64, output_dim=OUTPUT_DIM)
    enc_s.train()
    x_s = torch.randn(BATCH, 16)
    out_s = enc_s(x_s)
    assert out_s.shape == (BATCH, OUTPUT_DIM), f"got {out_s.shape}"
    # Verify gradient flows to both linear layers.
    out_s.sum().backward()
    assert enc_s.net[0].weight.grad is not None, "no grad on first Linear"
    assert enc_s.net[4].weight.grad is not None, "no grad on second Linear"
    enc_s.zero_grad()
    print(f"OK  {tuple(x_s.shape)} → {tuple(out_s.shape)}")

    # ------------------------------------------------------------------ 2
    print("Testing TextEncoder (downloads bert-base-uncased on first run) ...",
          end=" ", flush=True)
    enc_t = TextEncoder(output_dim=OUTPUT_DIM, freeze_bert=True)
    enc_t.eval()
    vocab_size = enc_t.bert.config.vocab_size
    input_ids   = torch.randint(0, vocab_size, (BATCH // 2, SEQ_LEN))
    attn_mask   = torch.ones(BATCH // 2, SEQ_LEN, dtype=torch.long)
    # Simulate padding: last 8 tokens of every second sample are padding.
    attn_mask[1::2, -8:] = 0
    with torch.no_grad():
        out_t = enc_t(input_ids, attn_mask)
    assert out_t.shape == (BATCH // 2, OUTPUT_DIM), f"got {out_t.shape}"
    # BERT weights must have no grad when frozen.
    for p in enc_t.bert.parameters():
        assert not p.requires_grad, "BERT param should be frozen"
    # Projection head must be trainable.
    for p in enc_t.proj.parameters():
        assert p.requires_grad, "proj param should be trainable"
    print(f"OK  ({BATCH // 2}, {SEQ_LEN}) → {tuple(out_t.shape)}")

    # ------------------------------------------------------------------ 3
    print("Testing BehavioralEncoder ...", end=" ", flush=True)
    enc_b = BehavioralEncoder(input_dim=2, hidden_dim=64, output_dim=OUTPUT_DIM, num_layers=2)
    enc_b.train()
    x_b = torch.randn(BATCH, 50, 2)
    out_b = enc_b(x_b)
    assert out_b.shape == (BATCH, OUTPUT_DIM), f"got {out_b.shape}"
    out_b.sum().backward()
    assert enc_b.proj.weight.grad is not None, "no grad on proj"
    print(f"OK  {tuple(x_b.shape)} → {tuple(out_b.shape)}")

    print("\nAll encoder smoke tests passed.")
    sys.exit(0)
