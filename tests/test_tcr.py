"""Unit tests for Temporal Context Routing (paper Eq. 1)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "ltx-core" / "src"))

from ltx_core.model.transformer.rope import compute_tcr_bias


def test_center_bias_is_zero() -> None:
    q = torch.tensor([[2.0]])
    k = torch.tensor([[[1.0, 3.0]]])  # c=2, r=1
    bias = compute_tcr_bias(q, k, beta=5.0, alpha=1.0)
    assert torch.allclose(bias, torch.zeros_like(bias), atol=1e-6)


def test_endpoint_retention_is_neg_beta_over_two() -> None:
    q = torch.tensor([[3.0]])
    k = torch.tensor([[[1.0, 3.0]]])  # endpoint, u=1
    bias = compute_tcr_bias(q, k, beta=5.0, alpha=1.0)
    assert torch.allclose(bias, torch.tensor([[[[-2.5]]]]), atol=1e-5)


def test_sentinel_tokens_receive_zero() -> None:
    q = torch.tensor([[1.5]])
    k = torch.tensor([[[-1.0, -1.0], [0.0, 2.0]]])
    bias = compute_tcr_bias(q, k, beta=5.0, alpha=1.0)
    assert bias[0, 0, 0, 0].item() == 0.0
    assert bias[0, 0, 0, 1].item() != 0.0


def test_formula_matches_paper() -> None:
    beta, alpha = 5.0, 1.0
    t, c, r = 0.5, 2.0, 1.5
    expected = -beta * alpha * (t - c) ** 2 / (2.0 * r**2)
    q = torch.tensor([[t]])
    k = torch.tensor([[[c - r, c + r]]])
    bias = compute_tcr_bias(q, k, beta=beta, alpha=alpha)
    assert math.isclose(bias.item(), expected, rel_tol=1e-6, abs_tol=1e-6)
