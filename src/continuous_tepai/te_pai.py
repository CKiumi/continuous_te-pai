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

    def sample_circuits(self, n: int) -> list[SampledCircuit]:
        """Batch-sample `n` circuits with vectorized NumPy calls.

        Much faster than calling `sample_circuit()` in a loop, because
        all random draws are done in bulk before any Python-level assembly.
        """
        if n <= 0:
            return []

        # 1. Gate counts for all circuits at once
        M_all = self._rng.poisson(self._Lambda, size=n)
        total = int(M_all.sum())

        if total == 0:
            return [
                SampledCircuit(rotations=(), weight=self._weight_prefactor)
                for _ in range(n)
            ]

        # 2. All times in one shot
        u = self._rng.random(total)
        all_times = np.interp(u, self._time_cdf, self._time_grid)
        # Sort per-circuit: since circuits are independent blocks in all_times,
        # sort each block individually below.

        # 3. All Pauli indices: vectorize coefficient evaluation on a time grid,
        # then interpolate and sample categorically.
        #
        # We evaluate |c_k(t)| on the same grid used for the time CDF,
        # interpolate per (k, t), normalize per t, and sample.
        L = self._ham.num_terms
        ts = self._time_grid
        coeff_grid = np.empty((L, ts.size))
        for k, c in enumerate(self._ham._coeffs):
            coeff_grid[k] = np.abs([c(t) for t in ts])

        # Interpolate |c_k(t)| at each sampled time t_m
        abs_coeffs = np.stack(
            [np.interp(all_times, ts, coeff_grid[k]) for k in range(L)],
            axis=1,
        )  # shape (total, L)

        # Sign of c_k(t_m) — needed later for the Δ angle direction
        sign_grid = np.empty((L, ts.size))
        for k, c in enumerate(self._ham._coeffs):
            sign_grid[k] = np.sign([c(t) for t in ts])
        signs = np.stack(
            [np.interp(all_times, ts, sign_grid[k]) for k in range(L)],
            axis=1,
        )

        row_sums = abs_coeffs.sum(axis=1, keepdims=True)
        probs = abs_coeffs / row_sums  # shape (total, L)

        # Vectorized categorical sampling from row-wise probabilities
        cdf = np.cumsum(probs, axis=1)
        rand = self._rng.random(total)[:, None]
        k_all = (rand < cdf).argmax(axis=1)  # shape (total,)

        # 4. Angle types for every gate
        ell_all = (self._rng.random(total) >= self._p_delta).astype(int)

        # 5. Assemble circuits
        circuits: list[SampledCircuit] = []
        paulis = self._ham.paulis
        offset = 0
        for M in M_all:
            if M == 0:
                circuits.append(
                    SampledCircuit(rotations=(), weight=self._weight_prefactor)
                )
                continue

            idx = slice(offset, offset + int(M))
            # Sort times within this circuit's block
            order = np.argsort(all_times[idx])
            k_c = k_all[idx][order]
            ell_c = ell_all[idx][order]
            start, stop = offset, offset + int(M)
            rows = np.arange(int(M))
            sign_c = signs[start:stop][rows, k_c]

            # Build rotations
            rotations = tuple(
                PauliRotation(
                    pauli=paulis[int(k_c[m])],
                    angle=(
                        float(sign_c[m]) * self._delta
                        if ell_c[m] == 0
                        else np.pi
                    ),
                )
                for m in range(int(M))
            )

            sign_product = float((-1.0) ** ell_c.sum())
            weight = sign_product * self._weight_prefactor
            circuits.append(SampledCircuit(rotations=rotations, weight=weight))

            offset += int(M)

        return circuits

    def _build_time_cdf(self, num_points: int = 4001) -> tuple[np.ndarray, np.ndarray]:
        T = self._total_time
        ts = np.linspace(0, T, num_points)
        densities = np.array([self._ham.l1_norm(t) for t in ts])
        cdf = integrate.cumulative_trapezoid(densities, ts, initial=0.0)
        cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
        return ts, cdf