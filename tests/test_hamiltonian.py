import numpy as np
from continuous_tepai import Hamiltonian, PauliString


def test_time_independent():
    H = Hamiltonian.time_independent([
        (PauliString("ZI"), 1.0),
        (PauliString("IX"), 0.5),
    ])
    assert H.num_qubits == 2
    assert H.num_terms == 2


def test_l1_norm_constant():
    H = Hamiltonian.time_independent([
        (PauliString("Z"), 1.0),
        (PauliString("X"), -2.0),
    ])
    assert H.l1_norm(0.0) == 3.0


def test_l1_norm_avg_constant():
    """For constant coefficients, the average should equal the instant value."""
    H = Hamiltonian.time_independent([
        (PauliString("Z"), 1.0),
        (PauliString("X"), -2.0),
    ])
    assert abs(H.l1_norm_avg(1.0) - 3.0) < 1e-10


def test_time_dependent():
    H = Hamiltonian([
        (PauliString("Z"), lambda t: np.cos(t)),
        (PauliString("X"), lambda t: np.sin(t)),
    ])
    coeffs = H.coefficients(0.0)
    assert abs(coeffs[0] - 1.0) < 1e-10   # cos(0) = 1
    assert abs(coeffs[1] - 0.0) < 1e-10   # sin(0) = 0


def test_time_dependent_l1_avg():
    """H(t) = cos(t) Z → ‖c‖_1_avg = (1/π) ∫_0^π |cos(t)| dt = 2/π."""
    H = Hamiltonian([(PauliString("Z"), lambda t: np.cos(t))])
    T = np.pi
    expected = 2.0 / np.pi
    assert abs(H.l1_norm_avg(T) - expected) < 1e-6


def test_mismatched_qubits():
    """Pauli strings with different qubit counts should raise."""
    try:
        Hamiltonian.time_independent([
            (PauliString("ZI"), 1.0),
            (PauliString("X"), 0.5),
        ])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass