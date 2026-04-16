# src/continuous_tepai/sampler.py

"""Core sampler for the Continuous TE-PAI protocol."""

from __future__ import annotations

import numpy as np
from scipy import integrate

from .hamiltonian import Hamiltonian, PauliString
from .circuit import PauliRotation, SampledCircuit

class ContinuousTEPAI:
    r"""Sampler for the Continuous TE-PAI protocol.

    Step 1: Draw gate count M ~ Poisson(Λ)
    Step 2: Draw M times from density f(t) = ‖c(t)‖₁ / (T · ‖c‖₁_avg)
    Step 3: For each t_m, draw k_m with Pr(k) = |c_k(t_m)| / ‖c(t_m)‖₁
    Step 4: Draw ℓ_m ∈ {0, 1} with Pr(ℓ=0) = 2 / (3 − cos Δ)
    Step 5: Build circuit U_ω = Π R_m and weight
            g_ω = (Π (-1)^ℓ_m) · exp(2 ‖c‖₁_avg T tan(Δ/2))
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        delta: float = 0.3,
        total_time: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not (0 < delta < np.pi / 2):
            raise ValueError(f"Δ must be in (0, π/2), got {delta}")
        if total_time <= 0:
            raise ValueError(f"T must be positive, got {total_time}")

        self._ham = hamiltonian
        self._delta = delta
        self._total_time = total_time
        self._rng = np.random.default_rng(seed)

        self._c1_avg: float = hamiltonian.l1_norm_avg(total_time)

        # Λ = csc(Δ)(3 − cos Δ) · ‖c‖₁_avg · T
        self._Lambda: float = (
            (1.0 / np.sin(delta))
            * (3.0 - np.cos(delta))
            * self._c1_avg
            * total_time
        )

        # p_Δ = 2 / (3 − cos Δ)
        self._p_delta: float = 2.0 / (3.0 - np.cos(delta))

        # Exponential prefactor of the weight
        self._weight_prefactor: float = np.exp(
            2.0 * self._c1_avg * total_time * np.tan(delta / 2.0)
        )

        # Build CDF table for time sampling
        self._time_grid, self._time_cdf = self._build_time_cdf()

    @property
    def expected_gate_count(self) -> float:
        return self._Lambda

    @property
    def p_delta(self) -> float:
        return self._p_delta

    @property
    def weight_prefactor(self) -> float:
        """exp(2 ‖c‖₁_avg T tan(Δ/2))."""
        return self._weight_prefactor

    def sample_gate_count(self) -> int:
        return int(self._rng.poisson(self._Lambda))

    def sample_times(self, M: int) -> np.ndarray:
        if M == 0:
            return np.array([])
        u = self._rng.random(M)
        times = np.interp(u, self._time_cdf, self._time_grid)
        times.sort()
        return times

    def sample_pauli_index(self, t: float) -> tuple[int, PauliString, float]:
        coeffs = self._ham.coefficients(t)
        abs_coeffs = np.abs(coeffs)
        probs = abs_coeffs / abs_coeffs.sum()
        k = int(self._rng.choice(self._ham.num_terms, p=probs))
        return k, self._ham.paulis[k], float(np.sign(coeffs[k]))

    def sample_angle(self, sign: float) -> tuple[float, int]:
        if self._rng.random() < self._p_delta:
            return sign * self._delta, 0
        return np.pi, 1

    def sample_circuit(self) -> SampledCircuit:
        """Steps 1–5: return one full sampled circuit with its weight."""
        M = self.sample_gate_count()
        if M == 0:
            return SampledCircuit(rotations=(), weight=self._weight_prefactor)

        times = self.sample_times(M)

        rotations: list[PauliRotation] = []
        sign_product = 1.0

        for t_m in times:
            _, pauli, sign = self.sample_pauli_index(t_m)
            angle, ell = self.sample_angle(sign)
            rotations.append(PauliRotation(pauli=pauli, angle=angle))
            sign_product *= (-1.0) ** ell

        weight = sign_product * self._weight_prefactor
        return SampledCircuit(rotations=tuple(rotations), weight=weight)

    def _build_time_cdf(self, num_points: int = 201) -> tuple[np.ndarray, np.ndarray]:
        T = self._total_time
        ts = np.linspace(0, T, num_points)
        densities = np.array([self._ham.l1_norm(t) for t in ts])
        cdf = integrate.cumulative_trapezoid(densities, ts, initial=0.0)
        cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
        return ts, cdf