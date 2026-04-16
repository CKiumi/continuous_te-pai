from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .hamiltonian import PauliString
from .te_pai import PauliRotation


@runtime_checkable
class Backend(Protocol):
    def expectation(
        self,
        rotations: Sequence[PauliRotation],
        observable: PauliString,
        num_qubits: int,
        *,
        shots: int | None = None,
        initial_state: str = "zero",
    ) -> float:
        """Compute ⟨ψ₀| U† O U |ψ₀⟩.

        initial_state : "zero" for |0...0⟩, "plus" for |+...+⟩.
        """
        ...