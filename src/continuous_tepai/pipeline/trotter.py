"""Trotter gate construction and execution.

Builds first-order Suzuki–Trotter rotations from a time-dependent
:class:`~continuous_tepai.hamiltonian.Hamiltonian` and provides efficient
execution paths for both MPS and statevector backends.

Two depth modes:
    **depth=1** — *linear*: ``N`` total steps distributed uniformly
    across the full time ``[0, T]``.

    **depth=2** — *adaptive quadratic*: ``N`` is the base step count for
    the first snapshot interval ``[0, dT]``.  The cumulative step count
    through snapshot *k* is ``N · k²``, so later intervals use more
    steps and keep second-order Trotter error roughly constant over
    increasing evolution time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..circuit import PauliRotation
from ..hamiltonian import Hamiltonian, PauliString


# ── gate construction ───────────────────────────────────────────────────

def build_trotter_rotations(
    ham: Hamiltonian,
    T: float,
    N: int,
    n_snapshots: int,
    depth: int = 1,
) -> list[list[PauliRotation]]:
    """Build Trotter rotation lists grouped by snapshot interval.

    Parameters
    ----------
    ham :
        Time-dependent Hamiltonian.
    T :
        Total simulation time.
    N :
        Trotter step count (total for depth=1, base for depth=2).
    n_snapshots :
        Number of equally-spaced measurement snapshots in ``(0, T]``.
    depth :
        1 = linear step count; 2 = adaptive quadratic.

    Returns
    -------
    list[list[PauliRotation]]
        A list of *n_snapshots* sublists, each containing the
        :class:`PauliRotation` objects for that snapshot interval.
    """
    if depth == 1:
        return _build_linear(ham, T, N, n_snapshots)
    if depth == 2:
        return _build_adaptive(ham, T, N, n_snapshots)
    raise ValueError(f"depth must be 1 or 2, got {depth}")


def _build_linear(
    ham: Hamiltonian, T: float, N: int, n_snapshots: int,
) -> list[list[PauliRotation]]:
    steps_per_snap = N // n_snapshots
    if steps_per_snap * n_snapshots != N:
        raise ValueError(
            f"N={N} must be divisible by n_snapshots={n_snapshots}"
        )
    dt = T / N
    rotations: list[list[PauliRotation]] = []
    for s in range(n_snapshots):
        snap_rots: list[PauliRotation] = []
        for step in range(steps_per_snap):
            t = (s * steps_per_snap + step) * dt
            coeffs = ham.coefficients(t)
            for k, pauli in enumerate(ham.paulis):
                angle = 2.0 * float(coeffs[k]) * dt
                if abs(angle) > 1e-15:
                    snap_rots.append(PauliRotation(pauli, angle))
        rotations.append(snap_rots)
    return rotations


def _build_adaptive(
    ham: Hamiltonian, T: float, N_base: int, n_snapshots: int,
) -> list[list[PauliRotation]]:
    dT = T / n_snapshots
    rotations: list[list[PauliRotation]] = []
    for k in range(1, n_snapshots + 1):
        N_cumul = N_base * k * k
        N_prev = N_base * (k - 1) * (k - 1)
        N_interval = N_cumul - N_prev
        t_start = (k - 1) * dT
        dt_step = dT / N_interval
        snap_rots: list[PauliRotation] = []
        for step in range(N_interval):
            t = t_start + step * dt_step
            coeffs = ham.coefficients(t)
            for ki, pauli in enumerate(ham.paulis):
                angle = 2.0 * float(coeffs[ki]) * dt_step
                if abs(angle) > 1e-15:
                    snap_rots.append(PauliRotation(pauli, angle))
        rotations.append(snap_rots)
    return rotations


# ── execution ───────────────────────────────────────────────────────────

def execute_trotter_mps(
    rotations_per_snapshot: list[list[PauliRotation]],
    observable: PauliString,
    n_qubits: int,
    initial_state: str,
    *,
    max_bond: int | None = None,
    cutoff: float = 1e-12,
) -> np.ndarray:
    """Execute Trotter evolution on a single MPS with snapshots.

    Creates one :class:`~quimb.tensor.CircuitMPS`, applies gates
    snapshot-by-snapshot, and measures the observable after each interval.
    This is *O(n_snapshots)* in MPS operations rather than *O(n_snapshots²)*
    for the cumulative-rotation approach.

    Returns
    -------
    np.ndarray
        Expectation values of shape ``(n_snapshots + 1,)``.  Index 0 is
        the initial-state measurement; indices ``1 … n_snapshots`` are
        measurements after each snapshot interval.
    """
    from ..backends.mps_backend import MPSBackend, _non_identity_sites

    be = MPSBackend(max_bond=max_bond, cutoff=cutoff)
    circ = be._make_circuit(n_qubits, initial_state)
    sites = _non_identity_sites(observable.label)

    results = [be._pauli_string_expectation(circ.psi, sites)]
    for snap_rots in rotations_per_snapshot:
        for rot in snap_rots:
            be._apply_rotation(circ, rot)
        results.append(be._pauli_string_expectation(circ.psi, sites))
    return np.array(results)


def execute_trotter_generic(
    rotations_per_snapshot: list[list[PauliRotation]],
    observable: PauliString,
    n_qubits: int,
    initial_state: str,
    backend,
) -> np.ndarray:
    """Execute Trotter evolution on any :class:`Backend`.

    Re-applies all rotations from scratch at each snapshot (statevector
    backends rebuild the state each call).  Correct but *O(n_snapshots²)*
    in circuit depth; acceptable for small systems.

    Returns
    -------
    np.ndarray
        Expectation values of shape ``(n_snapshots + 1,)``.
    """
    all_rots: list[PauliRotation] = []
    results = [
        backend.expectation([], observable, n_qubits, initial_state=initial_state)
    ]
    for snap_rots in rotations_per_snapshot:
        all_rots = all_rots + snap_rots
        ev = backend.expectation(
            all_rots, observable, n_qubits, initial_state=initial_state,
        )
        results.append(ev)
    return np.array(results)
