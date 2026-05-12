from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- Graph ---
    node_types: List[str] = field(
        default_factory=lambda: ["student", "course", "instructor", "resource"]
    )
    edge_types: List[str] = field(
        default_factory=lambda: [
            "enrolled_in", "collaborated_with", "accessed", "submitted_to"
        ]
    )

    # --- Encoders ---
    structured_input_dim: int = 64
    text_model_name: str = "bert-base-uncased"
    behavioral_input_dim: int = 32
    behavioral_seq_len: int = 50
    embed_dim: int = 128

    # --- HGNN ---
    num_hgnn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3

    # --- Training ---
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = 256

    # --- Multi-task loss weights ---
    lambda_grade: float = 0.4
    lambda_dropout: float = 0.4
    lambda_engagement: float = 0.2

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
