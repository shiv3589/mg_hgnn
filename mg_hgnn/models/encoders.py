import torch
import torch.nn as nn
from transformers import AutoModel

from config import Config


class StructuredEncoder(nn.Module):
    """Maps tabular/demographic features to embed_dim."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.structured_input_dim, cfg.embed_dim * 2),
            nn.LayerNorm(cfg.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, structured_input_dim)
        return self.net(x)


class TextEncoder(nn.Module):
    """Projects BERT CLS-token to embed_dim; optionally freezes BERT."""

    def __init__(self, cfg: Config, freeze_bert: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.text_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        bert_hidden = self.bert.config.hidden_size  # 768 for bert-base
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids / attention_mask: (N, seq_len)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        return_dict=True)
        cls = out.last_hidden_state[:, 0, :]  # (N, 768)
        return self.proj(cls)


class BehavioralEncoder(nn.Module):
    """Encodes variable-length activity log sequences via GRU -> embed_dim."""

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.embed_dim // 2  # bidirectional -> hidden*2 == embed_dim
        self.gru = nn.GRU(
            input_size=cfg.behavioral_input_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, behavioral_seq_len, behavioral_input_dim)
        # Use final hidden states of both directions concatenated
        _, h = self.gru(x)          # h: (4, N, hidden)  [2 layers * 2 dirs]
        fwd = h[-2]                 # last layer, forward
        bwd = h[-1]                 # last layer, backward
        out = torch.cat([fwd, bwd], dim=-1)   # (N, embed_dim)
        return self.drop(self.norm(out))
