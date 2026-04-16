from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable

from ..hamiltonian import PauliString
from ..te_pai import PauliRotation


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
    ) -> float: ...


def __getattr__(name: str):
    if name == "QiskitBackend":
        from .qiskit_backend import QiskitBackend
        return QiskitBackend
    if name == "QulacsBackend":
        from .qulacs_backend import QulacsBackend
        return QulacsBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Backend", "QiskitBackend", "QulacsBackend"]