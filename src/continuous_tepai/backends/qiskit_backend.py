"""Qiskit backend using the modern Primitives API (Qiskit ≥ 1.0)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
except ImportError as exc:
    raise ImportError(
        "Qiskit is required for QiskitBackend.\n"
        "Install with:  uv sync --extra qiskit"
    ) from exc

from ..hamiltonian import PauliString
from ..te_pai import PauliRotation


def _to_qiskit_pauli(ps: PauliString) -> Pauli:
    """Convert our PauliString to a Qiskit Pauli.

    Qiskit uses little-endian qubit ordering (rightmost = q0), so we
    reverse the label to match our big-endian convention.
    """
    return Pauli(ps.label[::-1])


class QiskitBackend:
    """Qiskit-based backend satisfying the Backend protocol.

    Parameters
    ----------
    estimator :
        Optional pre-configured EstimatorV2 (e.g. from qiskit_ibm_runtime).
        When None, uses exact Statevector simulation.
    """

    def __init__(self, *, estimator=None) -> None:
        self._external_estimator = estimator

    def expectation(
    self,
    rotations: Sequence[PauliRotation],
    observable: PauliString,
    num_qubits: int,
    *,
    shots: int | None = None,
    initial_state: str = "zero",
    ) -> float:
        qc = QuantumCircuit(num_qubits)

        # Prepare initial state
        if initial_state == "plus":
            qc.h(range(num_qubits))
        elif initial_state != "zero":
            raise ValueError(f"Unknown initial_state: {initial_state!r}")

        # Append TE-PAI rotations
        for rot in rotations:
            pauli = _to_qiskit_pauli(rot.pauli)
            gate = PauliEvolutionGate(pauli, time=rot.angle / 2)
            qc.append(gate, range(num_qubits))

        obs = SparsePauliOp(_to_qiskit_pauli(observable))

        if self._external_estimator is not None:
            job = self._external_estimator.run([(qc, obs)])
            return float(job.result()[0].data.evs[0])

        if shots is None:
            sv = Statevector.from_label("0" * num_qubits)
            sv = sv.evolve(qc)
            return float(np.real(sv.expectation_value(obs)))

        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator

        estimator = AerEstimator.from_backend(AerSimulator())
        estimator.options.default_shots = shots
        job = estimator.run([(qc, obs)])
        return float(job.result()[0].data.evs[0])