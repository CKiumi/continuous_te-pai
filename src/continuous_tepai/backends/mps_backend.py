"""MPS backend — matrix-product-state simulation via quimb.

Implements the same :class:`Backend` protocol as :class:`QulacsBackend` and
:class:`QiskitBackend`, so it is a drop-in replacement for running any
:class:`~continuous_tepai.SampledCircuit` through an MPS pipeline.

Supports:
    * Single-qubit Pauli rotations (X / Y / Z).
    * Two-qubit same-type Pauli rotations (XX / YY / ZZ).
    * Identity terms (absorbed as a trivial global phase).
    * Identity, single-site Pauli, and multi-site Pauli-string observables.

Anything else (e.g. mixed-type two-qubit rotations like ``XZ``) raises a
clear :class:`NotImplementedError` so silently-wrong answers are impossible.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import quimb as qu
    import quimb.tensor as qtn
except ImportError as exc:  # pragma: no cover - exercised only without quimb
    raise ImportError(
        "quimb is required for MPSBackend.\n"
        "Install with:  uv sync --extra mps"
    ) from exc

from ..hamiltonian import PauliString
from ..te_pai import PauliRotation


# Pauli rotation gate IDs accepted by quimb's ``CircuitMPS.apply_gate``.
_SINGLE_QUBIT_ROT = {"X": "rx", "Y": "ry", "Z": "rz"}
_TWO_QUBIT_ROT = {"XX": "rxx", "YY": "ryy", "ZZ": "rzz"}


def _non_identity_sites(label: str) -> list[tuple[int, str]]:
    """Return ``[(qubit_index, pauli_char), ...]`` for non-identity positions.

    Matches the qubit ordering used by the Qulacs and Qiskit backends:
    ``label[i]`` acts on qubit ``len(label) - 1 - i`` (big-endian label,
    little-endian qubit index).
    """
    n = len(label)
    return [(n - 1 - i, ch) for i, ch in enumerate(label) if ch != "I"]


class MPSBackend:
    """quimb-based MPS backend satisfying the :class:`Backend` protocol.

    Parameters
    ----------
    max_bond :
        Maximum bond dimension.  ``None`` (default) means no truncation —
        use this in parity tests against exact statevector backends.
    cutoff :
        Singular-value cutoff used during gate application.  Kept small
        so that, with ``max_bond=None``, the simulation is effectively
        exact up to floating-point noise.
    """

    def __init__(
        self,
        *,
        max_bond: int | None = None,
        cutoff: float = 1e-12,
    ) -> None:
        self.max_bond = max_bond
        self.cutoff = cutoff

    # ------------------------------------------------------------------
    # circuit construction
    # ------------------------------------------------------------------
    def _make_circuit(self, num_qubits: int, initial_state: str) -> qtn.CircuitMPS:
        kwargs: dict = {"cutoff": self.cutoff}
        if self.max_bond is not None:
            kwargs["max_bond"] = self.max_bond
        circ = qtn.CircuitMPS(num_qubits, **kwargs)

        if initial_state == "plus":
            for q in range(num_qubits):
                circ.apply_gate("H", qubits=[q])
        elif initial_state != "zero":
            raise ValueError(f"Unknown initial_state: {initial_state!r}")
        return circ

    # ------------------------------------------------------------------
    # gate application
    # ------------------------------------------------------------------
    def _apply_rotation(self, circ: qtn.CircuitMPS, rot: PauliRotation) -> None:
        sites = _non_identity_sites(rot.pauli.label)
        if not sites:
            # exp(-i θ/2 · I) is a global phase — no observable effect.
            return

        if len(sites) == 1:
            q, ch = sites[0]
            gate_id = _SINGLE_QUBIT_ROT[ch]
            circ.apply_gate(gate_id=gate_id, qubits=[q], params=[rot.angle])
            return

        if len(sites) == 2:
            (q1, c1), (q2, c2) = sites
            if c1 != c2:
                raise NotImplementedError(
                    "MPSBackend supports two-qubit Pauli rotations only for "
                    f"matching Pauli characters (XX, YY, ZZ); got {c1}{c2}."
                )
            key = c1 + c2
            if key not in _TWO_QUBIT_ROT:
                raise NotImplementedError(
                    f"MPSBackend cannot apply two-qubit rotation {key!r}."
                )
            # quimb's rxx/ryy/rzz are symmetric in their two qubits, so the
            # ordering here is irrelevant — but keep it deterministic.
            circ.apply_gate(
                gate_id=_TWO_QUBIT_ROT[key],
                qubits=[q1, q2],
                params=[rot.angle],
            )
            return

        raise NotImplementedError(
            "MPSBackend only supports single- and two-qubit Pauli rotations "
            f"(got {len(sites)}-body rotation on label {rot.pauli.label!r})."
        )

    # ------------------------------------------------------------------
    # observable evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _pauli_string_expectation(
        psi: qtn.MatrixProductState,
        sites: list[tuple[int, str]],
    ) -> float:
        """Evaluate ⟨ψ|P|ψ⟩ for a single Pauli-string observable P.

        Implemented uniformly by applying each Pauli factor of ``P`` as a
        single-site gate on a copy of ``ψ`` and taking the overlap with
        the original ``ψ``.  This is O(n · χ^3) in the bond dimension χ
        and works for any (single- or multi-site) Pauli observable.
        """
        if not sites:
            return 1.0

        psi_p = psi.copy()
        for q, ch in sites:
            psi_p.gate_(qu.pauli(ch), q, contract=True)
        overlap = psi.H @ psi_p
        return float(np.real(overlap))

    def expectation(
        self,
        rotations: Sequence[PauliRotation],
        observable: PauliString,
        num_qubits: int,
        *,
        shots: int | None = None,
        initial_state: str = "zero",
    ) -> float:
        """Return ⟨ψ|O|ψ⟩ with |ψ⟩ = U_rotations |initial_state⟩.

        Parameters
        ----------
        rotations :
            Ordered Pauli rotations forming the circuit U.
        observable :
            Pauli string observable O acting on ``num_qubits``.
        num_qubits :
            Number of qubits in the register.
        shots :
            Ignored — this backend is exact up to MPS truncation, so no
            sampling noise is injected.  Present only for protocol parity.
        initial_state :
            ``"zero"`` or ``"plus"``.
        """
        del shots  # unused — exact MPS evaluation

        circ = self._make_circuit(num_qubits, initial_state)
        for rot in rotations:
            self._apply_rotation(circ, rot)

        sites = _non_identity_sites(observable.label)
        return self._pauli_string_expectation(circ.psi, sites)

    def expectation_with_bond(
        self,
        rotations: Sequence[PauliRotation],
        observable: PauliString,
        num_qubits: int,
        *,
        initial_state: str = "zero",
    ) -> tuple[float, int]:
        """Like :meth:`expectation`, but also returns the MPS max bond dim."""
        circ = self._make_circuit(num_qubits, initial_state)
        for rot in rotations:
            self._apply_rotation(circ, rot)
        sites = _non_identity_sites(observable.label)
        ev = self._pauli_string_expectation(circ.psi, sites)
        return ev, int(circ.psi.max_bond())
