"""
Checkpoint merge methods. No SD.Next / Kohya dependencies — torch only.
All functions operate on individual tensors (key-by-key).

Two-model functions signature:  fn(a, b, alpha, **kwargs) -> Tensor
Three-model functions signature: fn(a, b, c, alpha, beta, density, **kwargs) -> Tensor
  where c is the base/anchor model.
"""
import torch
from torch import Tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_top_k(delta: Tensor, k: float) -> Tensor:
    """Keep the top-k fraction of delta by absolute value; zero the rest."""
    if k >= 1.0:
        return delta
    flat = delta.abs().flatten()
    if flat.numel() == 0:
        return delta
    threshold = flat.kthvalue(max(1, int(flat.numel() * (1.0 - k)))).values
    return delta * (delta.abs() >= threshold).float()


def _dare_preprocess(delta: Tensor, density: float) -> Tensor:
    """DARE: randomly zero (1-density) fraction of delta weights, then rescale."""
    if density >= 1.0:
        return delta
    mask = (torch.rand_like(delta) < density).float()
    return delta * mask / density


# ── Two-model methods ─────────────────────────────────────────────────────────

def weighted_sum(a: Tensor, b: Tensor, alpha: float = 0.5, **kwargs) -> Tensor:
    """Linear interpolation: (1-α)·A + α·B"""
    return (1.0 - alpha) * a + alpha * b


def slerp(a: Tensor, b: Tensor, alpha: float = 0.5, **kwargs) -> Tensor:
    """Spherical linear interpolation between two parameter tensors."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    an  = a_f / (a_f.norm() + 1e-8)
    bn  = b_f / (b_f.norm() + 1e-8)
    cos_theta = (an * bn).sum().clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return weighted_sum(a, b, alpha)
    sin_theta = torch.sin(theta)
    result = (torch.sin((1.0 - alpha) * theta) / sin_theta) * a_f + \
             (torch.sin(alpha * theta)          / sin_theta) * b_f
    return result.reshape(a.shape).to(a.dtype)


# ── Three-model methods (c = base/anchor) ─────────────────────────────────────

def add_difference(a: Tensor, b: Tensor, c: Tensor,
                   alpha: float = 0.5, **kwargs) -> Tensor:
    """A + α·(B − C)  — injects the B-vs-C difference into A."""
    return a + alpha * (b - c)


def ties_merge(a: Tensor, b: Tensor, c: Tensor,
               alpha: float = 0.5, beta: float = 0.7, **kwargs) -> Tensor:
    """
    TIES (arXiv:2306.01708): Trim-Elect-Sign-Disjoint merge.
    Trims low-magnitude deltas, elects final sign by majority vote,
    then averages only the deltas that agree with the elected sign.
    c = base, a and b = fine-tunes.
    """
    deltas = [_filter_top_k(m - c, beta) for m in (a, b)]
    signs  = [torch.sign(d) for d in deltas]

    final_sign = torch.sign(torch.stack(signs, dim=0).sum(dim=0))

    merged = torch.zeros_like(c)
    count  = torch.zeros_like(c)
    for s, d in zip(signs, deltas):
        mask    = (s == final_sign).float()
        merged += mask * d
        count  += mask

    merged = torch.nan_to_num(merged / count.clamp(min=1e-8))
    return c + alpha * merged


def dare_merge(a: Tensor, b: Tensor, c: Tensor,
               alpha: float = 0.5, density: float = 0.9, **kwargs) -> Tensor:
    """
    DARE: Drop And REscale before weighted merge.
    Applies random pruning + rescaling to each model's delta, then averages.
    """
    delta_a = _dare_preprocess(a - c, density)
    delta_b = _dare_preprocess(b - c, density)
    return c + alpha * (delta_a + delta_b) * 0.5


def dare_ties(a: Tensor, b: Tensor, c: Tensor,
              alpha: float = 0.5, beta: float = 0.7, density: float = 0.9,
              **kwargs) -> Tensor:
    """
    DARE + TIES combined: DARE pruning first, then TIES sign-elect merge.
    Best of both: sparse random pruning removes redundant parameters;
    TIES resolves sign conflicts between the two delta streams.
    """
    deltas = [_dare_preprocess(_filter_top_k(m - c, beta), density) for m in (a, b)]
    signs  = [torch.sign(d) for d in deltas]

    final_sign = torch.sign(torch.stack(signs, dim=0).sum(dim=0))

    merged = torch.zeros_like(c)
    count  = torch.zeros_like(c)
    for s, d in zip(signs, deltas):
        mask    = (s == final_sign).float()
        merged += mask * d
        count  += mask

    merged = torch.nan_to_num(merged / count.clamp(min=1e-8))
    return c + alpha * merged


# ── Registry ──────────────────────────────────────────────────────────────────

METHODS: dict[str, callable] = {
    "Weighted Sum":  weighted_sum,
    "Slerp":         slerp,
    "Add Difference":add_difference,
    "TIES":          ties_merge,
    "DARE":          dare_merge,
    "DARE+TIES":     dare_ties,
}

TWO_MODEL_METHODS   = {"Weighted Sum", "Slerp"}
THREE_MODEL_METHODS = {"Add Difference", "TIES", "DARE", "DARE+TIES"}

METHOD_DESCRIPTIONS = {
    "Weighted Sum":   "Linear blend: (1−α)·A + α·B.  Simple and predictable.",
    "Slerp":          "Spherical interpolation along the weight manifold.  Smoother than linear.",
    "Add Difference": "A + α·(B−C).  Injects the B-minus-C difference into A.  Requires 3 models.",
    "TIES":           "Trim low deltas, elect majority sign, disjoint merge.  Best for 2 fine-tunes "
                      "sharing the same base (C).  Requires 3 models.",
    "DARE":           "Random-prune + rescale deltas before averaging.  Reduces interference between "
                      "unrelated parameters.  Requires 3 models.",
    "DARE+TIES":      "DARE preprocessing followed by TIES sign-elect merge.  Strongest interference "
                      "rejection for 2-fine-tune merging.  Requires 3 models.",
}
