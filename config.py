"""Centralised hyperparameter configuration for MG-HGNN."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MGHGNNConfig:
    """All hyper-parameters and dataset settings for one training run.

    Attributes:
        hidden_dim: Shared embedding dimension *d* used by all encoders and
            graph layers.
        num_layers: Number of stacked MGHGNNLayer message-passing steps.
        num_heads: Number of attention heads inside each layer.
        dropout: Dropout probability applied after every layer activation.

        structured_input_dim: Raw dimensionality of tabular node features
            (set per dataset after loading).
        text_input_dim: Dimensionality of pre-computed text embeddings fed to
            TextEncoder (768 for BERT-base).
        behavioral_input_dim: Feature width of each time-step in the
            click-sequence (resource_id + log_clicks = 2 by default).
        behavioral_seq_len: Maximum sequence length kept per student.

        node_types: Ordered list of node type names.
        edge_types: List of (src_type, relation, dst_type) triples that define
            the heterogeneous graph schema.

        lr: Adam learning rate.
        weight_decay: L2 regularisation coefficient.
        epochs: Maximum training epochs.
        patience: Early-stopping patience (epochs without val improvement).
        batch_size: Mini-batch size used for the prediction head; the graph
            itself is processed full-batch.

        task: Downstream prediction task.  One of ``"grade"``,
            ``"dropout"``, or ``"engagement"``.
        num_classes: Output classes — ``1`` for regression (grade) or binary
            classification (dropout), ``>1`` for multi-class (engagement).

        dataset: Which dataset to load.  One of ``"oulad"`` or ``"moocc"``.
        data_dir: Root directory containing raw CSV / JSON data files.
        checkpoint_dir: Where to save the best model checkpoint.
        log_dir: Directory for TensorBoard / CSV training logs.

        seed: Global random seed for reproducibility.
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
    """

    # ------------------------------------------------------------------ model
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3

    # -------------------------------------------------------------- encoders
    structured_input_dim: int = 8     # updated at runtime from dataset
    text_input_dim: int = 768         # BERT-base hidden size
    behavioral_input_dim: int = 2     # (resource_idx, log_clicks)
    behavioral_seq_len: int = 50

    # ------------------------------------------------------------- graph schema
    node_types: List[str] = field(default_factory=lambda: [
        "student", "course", "instructor", "resource",
    ])
    edge_types: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("student",  "enrolled_in",       "course"),
        ("student",  "collaborated_with", "student"),
        ("student",  "accessed",          "resource"),
        ("student",  "submitted_to",      "course"),
    ])

    # ------------------------------------------------------------ training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 512

    # --------------------------------------------------------------- task
    task: str = "grade"       # "grade" | "dropout" | "engagement"
    num_classes: int = 1      # 1 = regression / binary; >1 = multi-class

    # --------------------------------------------------------------- data
    dataset: str = "oulad"    # "oulad" | "moocc"
    data_dir: str = os.path.join(os.path.dirname(__file__), "data", "raw")
    checkpoint_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir: str = os.path.join(os.path.dirname(__file__), "logs")

    # -------------------------------------------------------- multi-task weights
    lambda_grade:      float = 0.4
    lambda_dropout:    float = 0.4
    lambda_engagement: float = 0.2

    # ------------------------------------------------- cross-validation
    n_folds: int = 5

    # --------------------------------------------------------------- misc
    seed: int = 42
    device: str = "cuda"
    debug: bool = False

    def __post_init__(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
