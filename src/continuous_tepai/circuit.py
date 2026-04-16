from dataclasses import dataclass
from .hamiltonian import PauliString

@dataclass(frozen=True, slots=True)
class PauliRotation:
    """A Pauli rotation gate  R_P(θ) = exp(-i θ P / 2)."""

    pauli: PauliString
    angle: float


@dataclass(frozen=True, slots=True)
class SampledCircuit:
    """One random circuit sampled from Continuous TE-PAI.

    Attributes
    ----------
    rotations : ordered Pauli rotations forming U_ω.
    weight    : scalar weight g_ω for the unbiased estimator.
    """

    rotations: tuple[PauliRotation, ...]
    weight: float

    @property
    def gate_count(self) -> int:
        return len(self.rotations)