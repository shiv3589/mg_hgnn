import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ModalityGate(nn.Module):
    """Per-relation softmax gate over structured / text / behavioral embeddings.

    For each edge type r, a learned linear W_r maps the concatenated triple
    [h_s || h_t || h_b] -> 3 logits, whose softmax yields alpha = [α_s, α_t, α_b].
    The gated output is the alpha-weighted sum of the three input embeddings,
    keeping the output shape identical to each input modality.
    """

    def __init__(self, edge_types: List[str], embed_dim: int = 128):
        super().__init__()
        self.edge_types = edge_types
        self.embed_dim = embed_dim
        # string → integer index (used externally for logging / batching)
        self.edge_index: dict[str, int] = {r: i for i, r in enumerate(edge_types)}

        # One W_r per relation; bias=False keeps gate symmetric at init
        self.gates = nn.ModuleDict(
            {r: nn.Linear(3 * embed_dim, 3, bias=True) for r in edge_types}
        )

    def forward(
        self,
        edge_type: str,
        h_s: torch.Tensor,
        h_t: torch.Tensor,
        h_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_type: relation name, must be in self.edge_types
            h_s: structured embeddings  (N, embed_dim)
            h_t: text embeddings        (N, embed_dim)
            h_b: behavioral embeddings  (N, embed_dim)

        Returns:
            fused:  (N, embed_dim)  alpha-weighted combination
            alpha:  (N, 3)          per-sample gate weights [α_s, α_t, α_b]
        """
        concat = torch.cat([h_s, h_t, h_b], dim=-1)       # (N, 3*embed_dim)
        logits = self.gates[edge_type](concat)              # (N, 3)
        alpha = F.softmax(logits, dim=-1)                  # (N, 3) sums to 1

        alpha_s = alpha[:, 0:1]   # (N, 1)
        alpha_t = alpha[:, 1:2]
        alpha_b = alpha[:, 2:3]

        fused = alpha_s * h_s + alpha_t * h_t + alpha_b * h_b  # (N, embed_dim)
        return fused, alpha

    def get_gate_weights(self, edge_type: str) -> np.ndarray:
        """Return the mean gate vector for this relation under a neutral input.

        BUG (found while diagnosing Figure 2 / Table 6, 2026-08-11): this builds
        a synthetic identity-like probe (each modality slot a unit basis vector)
        and averages alpha over those embed_dim probe rows. That construction is
        not representative of real encoded embeddings (h_s/h_t/h_b have very
        different scale/covariance than one-hot rows) and empirically washes any
        learned per-relation differentiation back down to ~[0.33, 0.33, 0.33]
        REGARDLESS of training — verified by comparing this method's output
        against real-data alpha (see get_gate_weights_from_data) on a fully
        trained checkpoint: real data shows strong differentiation (e.g.
        enrolled_in -> [0.47, 0.14, 0.40]), this probe reports near-uniform for
        every relation on that same checkpoint. Kept for backward compatibility;
        prefer get_gate_weights_from_data for anything that reports learned
        behavior (figures, tables).
        """
        gate = self.gates[edge_type]
        d = self.embed_dim

        # Probe: embed_dim rows; each row has a 1 in one position per modality block
        # Row i  →  1 at position i (struct), 1 at position d+i (text), 1 at d*2+i (behav)
        probe = torch.zeros(d, 3 * d, device=next(gate.parameters()).device)
        idx = torch.arange(d)
        probe[idx, idx] = 1.0          # structured slot
        probe[idx, d + idx] = 1.0      # text slot
        probe[idx, 2 * d + idx] = 1.0  # behavioral slot

        with torch.no_grad():
            logits = gate(probe)                   # (d, 3)
            alpha = F.softmax(logits, dim=-1)      # (d, 3)
            mean_alpha = alpha.mean(dim=0)         # (3,)

        return mean_alpha.cpu().numpy()

    def get_gate_weights_from_data(
        self,
        edge_type: str,
        h_s: torch.Tensor,
        h_t: torch.Tensor,
        h_b: torch.Tensor,
    ) -> np.ndarray:
        """Return the mean gate vector for this relation on REAL encoded
        embeddings, i.e. the actual per-sample alpha the model produces during
        a forward pass, averaged over all N samples. This is what Figure 2 /
        Table 6 should report — see get_gate_weights docstring for why the
        synthetic-probe version is unreliable.
        """
        with torch.no_grad():
            _, alpha = self.forward(edge_type, h_s, h_t, h_b)   # (N, 3)
            mean_alpha = alpha.mean(dim=0)
        return mean_alpha.cpu().numpy()


# ----------------------------------------------------------------
# pytest block
# ----------------------------------------------------------------
if __name__ == "__pytest__" or __name__ == "__main__":
    import pytest

    EDGE_TYPES = ["enrolled_in", "collaborated_with", "accessed", "submitted_to"]
    EMBED_DIM = 128
    N = 16

    def make_gate():
        return ModalityGate(edge_types=EDGE_TYPES, embed_dim=EMBED_DIM)

    def random_inputs():
        return (
            torch.randn(N, EMBED_DIM),
            torch.randn(N, EMBED_DIM),
            torch.randn(N, EMBED_DIM),
        )

    def test_output_shape():
        gate = make_gate()
        h_s, h_t, h_b = random_inputs()
        fused, alpha = gate("enrolled_in", h_s, h_t, h_b)
        assert fused.shape == (N, EMBED_DIM), f"Bad fused shape: {fused.shape}"
        assert alpha.shape == (N, 3), f"Bad alpha shape: {alpha.shape}"

    def test_alpha_sums_to_one():
        gate = make_gate()
        h_s, h_t, h_b = random_inputs()
        _, alpha = gate("accessed", h_s, h_t, h_b)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(N), atol=1e-5), \
            f"Alpha row sums off: {sums}"

    def test_different_edge_types_differ():
        gate = make_gate()
        h_s, h_t, h_b = random_inputs()
        _, alpha1 = gate("enrolled_in", h_s, h_t, h_b)
        _, alpha2 = gate("collaborated_with", h_s, h_t, h_b)
        assert not torch.allclose(alpha1, alpha2), \
            "Different edge types should produce different alpha tensors"

    def test_get_gate_weights():
        gate = make_gate()
        for r in EDGE_TYPES:
            w = gate.get_gate_weights(r)
            assert w.shape == (3,), f"Bad shape for {r}: {w.shape}"
            assert abs(w.sum() - 1.0) < 1e-5, f"Weights don't sum to 1 for {r}: {w.sum()}"

    # Run all tests and report
    tests = [
        test_output_shape,
        test_alpha_sums_to_one,
        test_different_edge_types_differ,
        test_get_gate_weights,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASSED  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        raise SystemExit(1)
