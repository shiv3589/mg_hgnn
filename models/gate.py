"""Per-edge-type modality gate — core novel component of MG-HGNN.

The gate maintains one independent projection matrix W_r per relation type r
and produces a three-way softmax weight vector

    α(v, u, r) = softmax(W_r · [h_s(u) ∥ h_t(u) ∥ h_b(u)] + b_r)  ∈ ℝ³

that controls how much of the structured (s), textual (t), and behavioral (b)
information each relation type pulls from its source node during message
passing.  The gated source embedding is then:

    h_gated = α_s·h_s + α_t·h_t + α_b·h_b  ∈ ℝ^d

The key hypothesis is that α differs across relation types: enrolled_in edges
should weight structured curriculum metadata more heavily, while
collaborated_with edges should weight textual discussion features, and
accessed edges should weight behavioral clickstream features.  The gate learns
this automatically from data rather than requiring manual feature engineering.
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class ModalityGate(nn.Module):
    """Per-relation-type softmax gate over structured, textual, and behavioral
    modality embeddings.

    One :class:`~torch.nn.Linear` layer ``W_r: ℝ^{3d} → ℝ^3`` is
    maintained for every registered edge type.  All layers share the same
    architecture but have *independent* parameters, so each relation type
    learns its own modality preference.

    Args:
        embed_dim: Shared embedding dimension *d*.  All three modality
            tensors passed to :meth:`forward` must have this width.
        edge_types: Ordered list of relation-type name strings, e.g.
            ``["enrolled_in", "collaborated_with", "accessed",
            "submitted_to"]``.  The order determines the integer index used
            when calling :meth:`forward` with an ``int`` edge-type specifier.

    Raises:
        ValueError: If ``edge_types`` is empty.

    Example::

        gate = ModalityGate(embed_dim=128,
                            edge_types=["enrolled_in", "accessed"])
        h_gated = gate("enrolled_in", h_s, h_t, h_b)  # (N, 128)
    """

    #: Column labels for the α vector returned by :meth:`get_gate_weights`.
    MODALITY_NAMES: tuple = ("structured", "textual", "behavioral")
    NUM_MODALITIES: int = 3

    def __init__(self, embed_dim: int, edge_types: List[str]) -> None:
        super().__init__()
        if not edge_types:
            raise ValueError("edge_types must contain at least one relation type.")

        self.embed_dim = embed_dim
        self.edge_types: List[str] = list(edge_types)
        self._type_to_idx: Dict[str, int] = {
            et: i for i, et in enumerate(self.edge_types)
        }

        # One W_r per relation: ℝ^{3d} → ℝ^3
        # Stored in a ModuleList so PyTorch tracks parameters correctly.
        self.gate_layers = nn.ModuleList([
            nn.Linear(embed_dim * self.NUM_MODALITIES, self.NUM_MODALITIES, bias=True)
            for _ in edge_types
        ])

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def forward(
        self,
        edge_type: Union[str, int],
        h_s: Tensor,
        h_t: Tensor,
        h_b: Tensor,
    ) -> Tensor:
        """Compute the modality-gated source embedding for one edge relation.

        Calls :meth:`compute_alpha` internally, then returns the weighted
        combination ``α_s·h_s + α_t·h_t + α_b·h_b``.

        Args:
            edge_type: Relation type — either a registered string name (e.g.
                ``"enrolled_in"``) or a non-negative integer index into
                ``self.edge_types``.
            h_s: Structured embeddings of shape ``(N, d)``.
            h_t: Textual embeddings of shape ``(N, d)``.
            h_b: Behavioral embeddings of shape ``(N, d)``.

        Returns:
            Gated embedding of shape ``(N, d)``, a convex combination of the
            three input modalities weighted by the learned α.

        Raises:
            KeyError: Unknown string edge type.
            IndexError: Integer edge-type index out of range.
            TypeError: ``edge_type`` is neither str nor int.
        """
        alpha = self.compute_alpha(edge_type, h_s, h_t, h_b)   # (N, 3)
        return (
            alpha[:, 0:1] * h_s   # α_s · h_s
            + alpha[:, 1:2] * h_t  # α_t · h_t
            + alpha[:, 2:3] * h_b  # α_b · h_b
        )

    def compute_alpha(
        self,
        edge_type: Union[str, int],
        h_s: Tensor,
        h_t: Tensor,
        h_b: Tensor,
    ) -> Tensor:
        """Return the raw softmax gate weights α for every edge in the batch.

        This is the per-sample, per-edge-type attention vector.  Rows sum
        to 1.  Call this method directly when you need the α values for
        analysis or to verify the gate's behaviour during training.

        Args:
            edge_type: Relation type name or integer index.
            h_s: Structured embeddings ``(N, d)``.
            h_t: Textual embeddings ``(N, d)``.
            h_b: Behavioral embeddings ``(N, d)``.

        Returns:
            Float tensor of shape ``(N, 3)``.  Columns correspond to
            ``[α_structured, α_textual, α_behavioral]`` and sum to 1.
        """
        idx = self._resolve_idx(edge_type)
        concat = torch.cat([h_s, h_t, h_b], dim=-1)    # (N, 3d)
        logits = self.gate_layers[idx](concat)           # (N, 3)
        return torch.softmax(logits, dim=-1)             # (N, 3)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def get_gate_weights(
        self,
        edge_type: Union[str, int],
        h_s: Optional[Tensor] = None,
        h_t: Optional[Tensor] = None,
        h_b: Optional[Tensor] = None,
    ) -> np.ndarray:
        """Return the learned α vector for a given relation type as NumPy.

        Two modes of operation:

        **Data mode** — all three embedding tensors supplied:
            Computes α for every edge in the batch, then returns the
            **mean** across the batch.  Suitable for post-training analysis
            with representative samples.

        **Prior mode** — any embedding tensor is ``None``:
            Returns ``softmax(b_r)`` — the gate weights produced when all
            input embeddings are zero, which isolates the bias term ``b_r``
            learned for this relation type.  Useful for a quick look at the
            relation-specific modality prior without needing data.

        Args:
            edge_type: Relation type name or integer index.
            h_s: Optional structured embeddings ``(N, d)``.
            h_t: Optional textual embeddings ``(N, d)``.
            h_b: Optional behavioral embeddings ``(N, d)``.

        Returns:
            NumPy ``float32`` array of shape ``(3,)`` summing to 1.
            Indices map to ``ModalityGate.MODALITY_NAMES``:
            ``[α_structured, α_textual, α_behavioral]``.
        """
        idx = self._resolve_idx(edge_type)
        with torch.no_grad():
            if h_s is not None and h_t is not None and h_b is not None:
                alpha = self.compute_alpha(edge_type, h_s, h_t, h_b)
                result = alpha.mean(dim=0)
            else:
                # Prior: feed zeros so only the bias contributes to logits.
                device = self.gate_layers[idx].bias.device
                zeros = torch.zeros(1, self.embed_dim * self.NUM_MODALITIES,
                                    device=device)
                logits = self.gate_layers[idx](zeros)
                result = torch.softmax(logits, dim=-1).squeeze(0)
        return result.cpu().float().numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_idx(self, edge_type: Union[str, int]) -> int:
        """Translate a string or int edge-type specifier to a layer index.

        Args:
            edge_type: String name or non-negative integer.

        Returns:
            Integer index into ``self.gate_layers``.

        Raises:
            KeyError: Unknown string name.
            IndexError: Integer out of range ``[0, len(edge_types))``.
            TypeError: Neither str nor int.
        """
        if isinstance(edge_type, str):
            if edge_type not in self._type_to_idx:
                raise KeyError(
                    f"Unknown edge type '{edge_type}'. "
                    f"Registered types: {self.edge_types}"
                )
            return self._type_to_idx[edge_type]

        if isinstance(edge_type, int):
            n = len(self.edge_types)
            if not 0 <= edge_type < n:
                raise IndexError(
                    f"Integer edge-type index {edge_type} is out of range "
                    f"[0, {n}).  Known types: {self.edge_types}"
                )
            return edge_type

        raise TypeError(
            f"edge_type must be str or int, got {type(edge_type).__name__!r}."
        )

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_relations={len(self.edge_types)}, "
            f"edge_types={self.edge_types}"
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestModalityGate(unittest.TestCase):
    """Unit tests for ModalityGate covering shape, softmax constraint, and
    per-relation independence."""

    # Four canonical relation types from the MG-HGNN schema
    EDGE_TYPES = [
        "enrolled_in",
        "collaborated_with",
        "accessed",
        "submitted_to",
    ]
    D = 32   # embedding dimension
    N = 16   # batch / edge count

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.gate = ModalityGate(embed_dim=self.D, edge_types=self.EDGE_TYPES)
        self.h_s = torch.randn(self.N, self.D)
        self.h_t = torch.randn(self.N, self.D)
        self.h_b = torch.randn(self.N, self.D)

    # ------------------------------------------------------------------
    # Test 1 — output shape is (N, d)
    # ------------------------------------------------------------------

    def test_output_shape_string_keys(self) -> None:
        """forward() returns (N, d) for every registered string edge type."""
        for et in self.EDGE_TYPES:
            with self.subTest(edge_type=et):
                out = self.gate(et, self.h_s, self.h_t, self.h_b)
                self.assertEqual(
                    out.shape, (self.N, self.D),
                    msg=f"Expected ({self.N}, {self.D}), got {tuple(out.shape)}",
                )

    def test_output_shape_int_keys(self) -> None:
        """forward() accepts integer indices and returns (N, d)."""
        for i in range(len(self.EDGE_TYPES)):
            with self.subTest(edge_type_idx=i):
                out = self.gate(i, self.h_s, self.h_t, self.h_b)
                self.assertEqual(out.shape, (self.N, self.D))

    # ------------------------------------------------------------------
    # Test 2 — gate weights sum to 1.0 for every sample
    # ------------------------------------------------------------------

    def test_alpha_sums_to_one_per_sample(self) -> None:
        """α_s + α_t + α_b == 1 for every sample in every edge type."""
        for et in self.EDGE_TYPES:
            with self.subTest(edge_type=et):
                alpha = self.gate.compute_alpha(et, self.h_s, self.h_t, self.h_b)
                self.assertEqual(alpha.shape, (self.N, 3))

                row_sums = alpha.sum(dim=-1)           # (N,)
                max_dev = (row_sums - 1.0).abs().max().item()
                self.assertAlmostEqual(
                    max_dev, 0.0, places=5,
                    msg=(
                        f"Gate weights don't sum to 1 for '{et}'. "
                        f"Max |sum−1| = {max_dev:.2e}"
                    ),
                )

    def test_get_gate_weights_prior_sums_to_one(self) -> None:
        """Prior gate weights (no input) also sum to 1."""
        for et in self.EDGE_TYPES:
            with self.subTest(edge_type=et):
                w = self.gate.get_gate_weights(et)
                self.assertEqual(w.shape, (3,))
                self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_get_gate_weights_data_mode_sums_to_one(self) -> None:
        """Data-mode gate weights (mean over batch) also sum to 1."""
        for et in self.EDGE_TYPES:
            with self.subTest(edge_type=et):
                w = self.gate.get_gate_weights(et, self.h_s, self.h_t, self.h_b)
                self.assertEqual(w.shape, (3,))
                self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    # ------------------------------------------------------------------
    # Test 3 — different edge types produce different α vectors
    # ------------------------------------------------------------------

    def test_different_edge_types_produce_different_alpha(self) -> None:
        """Every pair of edge types must yield distinguishably different mean α.

        With distinct random initialisations (seed=42) the four W_r matrices
        map the same input to different logits, so their mean softmax outputs
        must differ.  We use a loose atol=1e-3 to stay robust across hardware.
        """
        mean_alphas = {
            et: self.gate.compute_alpha(et, self.h_s, self.h_t, self.h_b)
                       .mean(dim=0)
                       .detach()
            for et in self.EDGE_TYPES
        }
        types = list(mean_alphas)
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                a, b = mean_alphas[types[i]], mean_alphas[types[j]]
                are_same = torch.allclose(a, b, atol=1e-3)
                self.assertFalse(
                    are_same,
                    msg=(
                        f"'{types[i]}' and '{types[j]}' produced the same "
                        f"mean α:\n  {types[i]}: {a.numpy()}\n"
                        f"  {types[j]}: {b.numpy()}"
                    ),
                )

    def test_prior_weights_differ_across_relation_types(self) -> None:
        """Bias-only (prior) gate weights must differ across relation types."""
        priors = {et: self.gate.get_gate_weights(et) for et in self.EDGE_TYPES}
        types = list(priors)
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                self.assertFalse(
                    np.allclose(priors[types[i]], priors[types[j]], atol=1e-4),
                    msg=(
                        f"Prior weights for '{types[i]}' and '{types[j]}' "
                        f"are unexpectedly identical."
                    ),
                )

    # ------------------------------------------------------------------
    # Correctness / consistency tests
    # ------------------------------------------------------------------

    def test_forward_equals_manual_combination(self) -> None:
        """forward() output must equal α_s·h_s + α_t·h_t + α_b·h_b exactly."""
        for et in self.EDGE_TYPES:
            with self.subTest(edge_type=et):
                alpha = self.gate.compute_alpha(et, self.h_s, self.h_t, self.h_b)
                expected = (
                    alpha[:, 0:1] * self.h_s
                    + alpha[:, 1:2] * self.h_t
                    + alpha[:, 2:3] * self.h_b
                )
                actual = self.gate(et, self.h_s, self.h_t, self.h_b)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-6),
                    msg=f"forward() mismatch for '{et}'",
                )

    def test_int_and_string_keys_are_equivalent(self) -> None:
        """Integer and string specifiers for the same relation give identical output."""
        for i, et in enumerate(self.EDGE_TYPES):
            with self.subTest(edge_type=et):
                out_str = self.gate(et, self.h_s, self.h_t, self.h_b)
                out_int = self.gate(i,  self.h_s, self.h_t, self.h_b)
                self.assertTrue(torch.allclose(out_str, out_int))

    def test_gradient_flows_only_to_used_gate_layer(self) -> None:
        """Backprop touches W_r for the used relation but not the others."""
        target_et = "enrolled_in"
        target_idx = self.gate._resolve_idx(target_et)

        self.gate.zero_grad()
        out = self.gate(target_et, self.h_s, self.h_t, self.h_b)
        out.sum().backward()

        self.assertIsNotNone(
            self.gate.gate_layers[target_idx].weight.grad,
            msg=f"Expected gradient on gate_layers[{target_idx}] ('{target_et}')",
        )
        for i, layer in enumerate(self.gate.gate_layers):
            if i != target_idx:
                self.assertIsNone(
                    layer.weight.grad,
                    msg=(
                        f"Unexpected gradient on gate_layers[{i}] "
                        f"('{self.EDGE_TYPES[i]}') — only '{target_et}' was used."
                    ),
                )

    # ------------------------------------------------------------------
    # Error-handling tests
    # ------------------------------------------------------------------

    def test_unknown_string_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.gate("nonexistent_relation", self.h_s, self.h_t, self.h_b)

    def test_out_of_range_int_raises_index_error(self) -> None:
        with self.assertRaises(IndexError):
            self.gate(99, self.h_s, self.h_t, self.h_b)

    def test_wrong_type_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.gate(3.14, self.h_s, self.h_t, self.h_b)  # type: ignore[arg-type]

    def test_empty_edge_types_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ModalityGate(embed_dim=16, edge_types=[])


if __name__ == "__main__":
    unittest.main(verbosity=2)
