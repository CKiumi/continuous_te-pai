"""Qulacs backend — fast C++ statevector simulation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from qulacs import QuantumCircuit, QuantumState, Observable as QulacsObservable
    from qulacs.gate import PauliRotation as QulacsPauliRotation
except ImportError as exc:
    raise ImportError(
        "Qulacs is required for QulacsBackend.\n"
        "Install with:  uv sync --extra qulacs"
    ) from exc

from ..hamiltonian import PauliString
from ..te_pai import PauliRotation


# Qulacs uses integer Pauli IDs: I=0, X=1, Y=2, Z=3
_PAULI_ID = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _non_identity_targets(label: str) -> tuple[list[int], list[int]]:
    """Return (qubit_indices, pauli_ids) for all non-identity positions.

    Qulacs uses little-endian ordering: index 0 = rightmost character.
    """
    n = len(label)
    indices, ids = [], []
    for i, ch in enumerate(label):
        if ch != "I":
            indices.append(n - 1 - i)          # big-endian → little-endian
            ids.append(_PAULI_ID[ch])
    return indices, ids


class QulacsBackend:
    """Qulacs-based backend satisfying the Backend protocol.

    Much faster than the Qiskit backend for inner sampling loops
    because gate application is near-zero overhead C++.
    """

    def expectation(
        self,
        rotations: Sequence[PauliRotation],
        observable: PauliString,
        num_qubits: int,
        *,
        shots: int | None = None,
        initial_state: str = "zero",
    ) -> float:
        state = QuantumState(num_qubits)
        state.set_zero_state()

        # Prepare initial state
        if initial_state == "plus":
            qc_prep = QuantumCircuit(num_qubits)
            for q in range(num_qubits):
                qc_prep.add_H_gate(q)
            qc_prep.update_quantum_state(state)
        elif initial_state != "zero":
            raise ValueError(f"Unknown initial_state: {initial_state!r}")

        # Apply TE-PAI rotations
        qc = QuantumCircuit(num_qubits)
        for rot in rotations:
            indices, ids = _non_identity_targets(rot.pauli.label)
            if not indices:
                continue  # all-identity rotation = global phase, skip
            # Qulacs: PauliRotation(target_list, pauli_id_list, angle)
            # applies exp(-i * angle/2 * P) — same convention as ours
            qc.add_gate(QulacsPauliRotation(indices, ids, rot.angle))
        qc.update_quantum_state(state)

        # Build and evaluate the observable
        obs = QulacsObservable(num_qubits)
        obs_indices, obs_ids = _non_identity_targets(observable.label)
        pauli_term = " ".join(
            f"{'IXYZ'[pid]} {q}" for q, pid in zip(obs_indices, obs_ids)
        )
        if not pauli_term:
            return 1.0  # identity observable
        obs.add_operator(1.0, pauli_term)

        return float(obs.get_expectation_value(state))