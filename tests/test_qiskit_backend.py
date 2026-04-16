import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from continuous_tepai import PauliString
from continuous_tepai.te_pai import PauliRotation
from continuous_tepai.backends import QiskitBackend


def test_identity_circuit():
    """No rotations → ⟨Z⟩ on |0⟩ = 1."""
    backend = QiskitBackend()
    ev = backend.expectation([], PauliString("Z"), num_qubits=1)
    assert abs(ev - 1.0) < 1e-10


def test_x_rotation_on_z():
    """R_X(π) |0⟩ = -i|1⟩ → ⟨Z⟩ = -1."""
    backend = QiskitBackend()
    rot = PauliRotation(PauliString("X"), angle=np.pi)
    ev = backend.expectation([rot], PauliString("Z"), num_qubits=1)
    assert abs(ev - (-1.0)) < 1e-10


def test_x_rotation_half():
    """R_X(π/2) |0⟩ → ⟨Z⟩ = cos(π/2) = 0, ⟨Y⟩ = -sin(π/2) = -1."""
    backend = QiskitBackend()
    rot = PauliRotation(PauliString("X"), angle=np.pi / 2)
    assert abs(backend.expectation([rot], PauliString("Z"), 1)) < 1e-10
    assert abs(backend.expectation([rot], PauliString("Y"), 1) - (-1.0)) < 1e-10


def test_two_qubit():
    """R_X(π) on qubit 0 → ⟨ZI⟩ = -1, ⟨IZ⟩ = 1."""
    backend = QiskitBackend()
    rot = PauliRotation(PauliString("XI"), angle=np.pi)
    assert abs(backend.expectation([rot], PauliString("ZI"), 2) - (-1.0)) < 1e-10
    assert abs(backend.expectation([rot], PauliString("IZ"), 2) - 1.0) < 1e-10