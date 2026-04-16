import numpy as np
import pytest

pytest.importorskip("qulacs")

from continuous_tepai import PauliString
from continuous_tepai.te_pai import PauliRotation
from continuous_tepai.backends import QulacsBackend


def test_identity_circuit():
    backend = QulacsBackend()
    ev = backend.expectation([], PauliString("Z"), num_qubits=1)
    assert abs(ev - 1.0) < 1e-10


def test_x_rotation_pi():
    """R_X(π)|0⟩ = -i|1⟩ → ⟨Z⟩ = -1."""
    backend = QulacsBackend()
    rot = PauliRotation(PauliString("X"), angle=np.pi)
    ev = backend.expectation([rot], PauliString("Z"), num_qubits=1)
    assert abs(ev - (-1.0)) < 1e-10


def test_plus_initial_state():
    """|+⟩ → ⟨X⟩ = 1."""
    backend = QulacsBackend()
    ev = backend.expectation([], PauliString("X"), num_qubits=1, initial_state="plus")
    assert abs(ev - 1.0) < 1e-10


def test_two_qubit_ordering():
    """R_X(π) on the first character (qubit 0 big-endian) → ⟨ZI⟩ = -1."""
    backend = QulacsBackend()
    rot = PauliRotation(PauliString("XI"), angle=np.pi)
    assert abs(backend.expectation([rot], PauliString("ZI"), 2) - (-1.0)) < 1e-10
    assert abs(backend.expectation([rot], PauliString("IZ"), 2) - 1.0) < 1e-10