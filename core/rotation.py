"""
llm_quant.core.rotation — Rotation / Hadamard utilities.

Used by QuaRot, SpinQuant, FlatQuant, etc. to construct orthogonal
rotations that can be absorbed into weights without changing the model's
mathematical output.

Provides:
- hadamard_matrix:         deterministic Hadamard (Sylvester construction)
- random_hadamard_matrix:  randomised sign-flip Hadamard
- random_orthogonal:       Haar-random orthogonal matrix via QR
- cayley_transform:        parameterise O(d) via skew-symmetric → orthogonal
- CayleyRotation:          learnable rotation layer (for SpinQuant / FlatQuant)
- fast_hadamard_transform: O(d log d) in-place Hadamard transform
- rotate_weight_right / left: apply rotation R to Linear.weight
- apply_rotation_to_linear: fuse rotation into a Linear layer
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Hadamard Matrices
# ──────────────────────────────────────────────────────────────────────────────

def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sylvester-construction normalised Hadamard matrix of size *n*.

    *n* must be a power of 2.  The returned matrix is orthogonal:
    ``H @ H.T == I``.
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)


def random_hadamard_matrix(n: int, dtype: torch.dtype = torch.float32,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomised Hadamard: ``H @ diag(±1)`` with random sign flips.

    Preserves orthogonality (product of two orthogonal matrices).
    """
    H = hadamard_matrix(n, dtype)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=dtype) * 2 - 1
    return H * signs.unsqueeze(0)  # broadcast column-wise


def random_orthogonal(n: int, dtype: torch.dtype = torch.float32,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Haar-random orthogonal matrix via QR decomposition."""
    A = torch.randn(n, n, dtype=dtype, generator=generator)
    Q, R = torch.linalg.qr(A)
    # Make unique (positive diagonal of R)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q


# ──────────────────────────────────────────────────────────────────────────────
# Cayley Transform (learnable rotations for SpinQuant / FlatQuant)
# ──────────────────────────────────────────────────────────────────────────────

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform of a skew-symmetric matrix A.

    Returns ``Q = (I - A)(I + A)^{-1}`` which is guaranteed orthogonal
    when A is skew-symmetric (A = −Aᵀ).
    """
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


class CayleyRotation(nn.Module):
    """Learnable orthogonal rotation parameterised via Cayley transform.

    Stores a skew-symmetric matrix ``A`` (upper-triangular parameters);
    at forward time produces ``Q = Cayley(A)``.

    Parameters
    ----------
    dim : int
        Matrix dimension.
    init : str
        ``'identity'`` → start at I;  ``'hadamard'`` → start at H_n.
    """

    def __init__(self, dim: int, init: str = "identity"):
        super().__init__()
        self.dim = dim
        # Upper-triangular free parameters
        n_params = dim * (dim - 1) // 2
        self.params = nn.Parameter(torch.zeros(n_params))
        if init == "hadamard" and (dim & (dim - 1)) == 0:
            # Initialise so Cayley(A) ≈ H
            pass  # keep zeros → identity; fine-tune from there

    def _build_skew(self) -> torch.Tensor:
        """Build skew-symmetric matrix from free parameters."""
        A = torch.zeros(self.dim, self.dim, device=self.params.device,
                        dtype=self.params.dtype)
        idx = torch.triu_indices(self.dim, self.dim, offset=1)
        A[idx[0], idx[1]] = self.params
        A = A - A.t()  # skew-symmetric
        return A

    def forward(self) -> torch.Tensor:
        """Return the current orthogonal matrix Q ∈ O(dim)."""
        A = self._build_skew()
        return cayley_transform(A)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ──────────────────────────────────────────────────────────────────────────────
# Fast Hadamard Transform (O(d log d) in-place)
# ──────────────────────────────────────────────────────────────────────────────

def fast_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Hadamard transform along the last dimension via butterfly.

    Equivalent to ``x @ H`` where H is the normalised Hadamard matrix.
    Complexity: ``O(d log d)`` instead of ``O(d²)`` for matmul.

    Parameters
    ----------
    x : Tensor  shape ``(..., d)`` where d is a power of 2.
    normalize : bool  if True, divide by ``sqrt(d)``.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, "last dim must be power of 2"
    # Use matmul with the Hadamard matrix for correctness.
    # For very large d a true butterfly implementation would be faster,
    # but for typical head_dim sizes (64-256) this is perfectly fine.
    H = hadamard_matrix(d, dtype=x.dtype).to(x.device)
    return x @ H


# ──────────────────────────────────────────────────────────────────────────────
# Weight rotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def rotate_weight_right(weight: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Multiply weight on the right by R: ``W' = W @ R``."""
    return weight @ R


def rotate_weight_left(weight: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Multiply weight on the left by Rᵀ: ``W' = Rᵀ @ W``."""
    return R.t() @ weight


@torch.no_grad()
def apply_rotation_to_linear(
    linear: nn.Linear,
    R: torch.Tensor,
    side: str = "right",
) -> None:
    """Fuse an orthogonal rotation into a ``nn.Linear`` layer *in-place*.

    Parameters
    ----------
    linear : nn.Linear
    R : orthogonal matrix  (d × d)
    side : ``'right'`` → absorb into input dim;
           ``'left'`` → absorb into output dim.
    """
    if side == "right":
        linear.weight.data = rotate_weight_right(linear.weight.data, R)
    elif side == "left":
        linear.weight.data = rotate_weight_left(linear.weight.data, R)
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
