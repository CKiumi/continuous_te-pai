# src/continuous_tepai/hamiltonian.py

"""Time-dependent Hamiltonian  H(t) = Σ_k c_k(t) P_k."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import warnings

import numpy as np
from scipy import integrate
from scipy.integrate import IntegrationWarning

# A coefficient function  c_k : [0, T] → ℝ
CoefficientFn = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class PauliString:
    """A Pauli string on *n* qubits.

    Stored as a label using characters {'I','X','Y','Z'}.
    Example: PauliString("XIZI") represents X⊗I⊗Z⊗I.
    """

    label: str

    def __post_init__(self) -> None:
        if not all(ch in "IXYZ" for ch in self.label):
            raise ValueError(f"Invalid Pauli label: {self.label!r}")

    @property
    def num_qubits(self) -> int:
        return len(self.label)


class Hamiltonian:
    r"""Time-dependent Pauli Hamiltonian  H(t) = Σ_k c_k(t) P_k.

    Parameters
    ----------
    terms : sequence of (PauliString, CoefficientFn) pairs.
        Each pair (P_k, c_k) defines one term, where c_k(t) returns
        a real scalar for any t ∈ [0, T].

    Examples
    --------
    Transverse-field Ising on 2 qubits::

        H = Hamiltonian([
            (PauliString("ZZ"), lambda t: 1.0),
            (PauliString("XI"), lambda t: 0.5),
            (PauliString("IX"), lambda t: 0.5),
        ])

    Time-dependent drive::

        H = Hamiltonian([
            (PauliString("ZI"), lambda t: np.cos(t)),
            (PauliString("IX"), lambda t: np.sin(t)),
        ])
    """

    def __init__(self, terms: Sequence[tuple[PauliString, CoefficientFn]]) -> None:
        if not terms:
            raise ValueError("Hamiltonian must contain at least one term.")
        self._paulis: tuple[PauliString, ...] = tuple(p for p, _ in terms)
        self._coeffs: tuple[CoefficientFn, ...] = tuple(c for _, c in terms)

        nq = {p.num_qubits for p in self._paulis}
        if len(nq) != 1:
            raise ValueError(
                f"All Pauli strings must act on the same number of qubits; got {nq}"
            )

    @property
    def num_qubits(self) -> int:
        return self._paulis[0].num_qubits

    @property
    def num_terms(self) -> int:
        """L = number of Pauli terms."""
        return len(self._paulis)

    @property
    def paulis(self) -> tuple[PauliString, ...]:
        return self._paulis

    def coefficients(self, t: float) -> np.ndarray:
        """Return the vector [c_1(t), ..., c_L(t)]."""
        return np.array([c(t) for c in self._coeffs])

    def l1_norm(self, t: float) -> float:
        """‖c(t)‖_1 = Σ_k |c_k(t)|."""
        return float(np.sum(np.abs(self.coefficients(t))))

    def l1_norm_avg(self, T: float, *, quad_points: int = 201) -> float:
        r"""Time-averaged ℓ₁-norm:  (1/T) ∫_0^T ‖c(t)‖_1 dt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            val, _ = integrate.quad(self.l1_norm, 0.0, T, limit=quad_points)
        return val / T

    @classmethod
    def time_independent(
        cls,
        terms: Sequence[tuple[PauliString, float]],
    ) -> Hamiltonian:
        """Build from constant coefficients  H = Σ_k a_k P_k.

        Example::

            H = Hamiltonian.time_independent([
                (PauliString("ZZ"), 1.0),
                (PauliString("XI"), 0.5),
            ])
        """
        return cls([(p, _const(a)) for p, a in terms])

    @classmethod
    def from_local_terms(
        cls,
        num_qubits: int,
        terms: Sequence[tuple[str, Sequence[int], CoefficientFn]],
    ) -> Hamiltonian:
        """Build from local Pauli terms.

        Each term is (gate_string, qubit_indices, coefficient_fn).

        Example:
            H = Hamiltonian.from_local_terms(3, [
                ("XX", [0, 1], lambda t: 1.0),
                ("Z",  [2],    lambda t: 0.5),
            ])
        """
        full_terms: list[tuple[PauliString, CoefficientFn]] = []
        for gate, qubits, coef in terms:
            label = ["I"] * num_qubits
            for g, q in zip(gate, qubits, strict=True):
                label[q] = g
            full_terms.append((PauliString("".join(label)), coef))
        return cls(full_terms)


def _const(a: float) -> CoefficientFn:
    """Return a closure that always returns *a*."""
    def _fn(t: float) -> float:
        return a
    return _fn