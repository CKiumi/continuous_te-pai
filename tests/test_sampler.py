import numpy as np
import pytest
from continuous_tepai import ContinuousTEPAI, Hamiltonian, PauliString


def test_lambda_formula():
    delta = 0.3
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=delta, total_time=1.0)

    expected = (1.0 / np.sin(delta)) * (3.0 - np.cos(delta))
    assert abs(sampler.expected_gate_count - expected) < 1e-10


def test_gate_count_mean():
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=0)

    samples = [sampler.sample_gate_count() for _ in range(10000)]
    assert abs(np.mean(samples) - sampler.expected_gate_count) < 0.2


def test_invalid_delta():
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    with pytest.raises(ValueError):
        ContinuousTEPAI(H, delta=0.0)


# -- time sampling tests --

def test_times_sorted():
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=2.0, seed=42)

    times = sampler.sample_times(20)
    assert np.all(times[:-1] <= times[1:])


def test_times_in_range():
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    T = 3.0
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=T, seed=42)

    times = sampler.sample_times(100)
    assert np.all(times >= 0)
    assert np.all(times <= T)


def test_times_zero_gates():
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=42)

    times = sampler.sample_times(0)
    assert len(times) == 0


def test_times_density_time_dependent():
    """For H(t) = cos(t) Z, times should cluster near t=0 and t=π."""
    H = Hamiltonian([(PauliString("Z"), lambda t: np.cos(t))])
    T = np.pi
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=T, seed=0)

    times = sampler.sample_times(10000)
    near_zero = np.sum(times < 0.5)
    near_mid = np.sum((times > 1.2) & (times < 1.9))
    assert near_zero > near_mid


# -- pauli index sampling tests --

def test_pauli_index_valid():
    H = Hamiltonian.time_independent([
        (PauliString("ZZ"), 1.0),
        (PauliString("XI"), 0.5),
        (PauliString("IX"), 0.5),
    ])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=42)

    for _ in range(50):
        k, pauli, sign = sampler.sample_pauli_index(0.0)
        assert 0 <= k < H.num_terms
        assert pauli == H.paulis[k]
        assert sign in (-1.0, 1.0)


def test_pauli_index_distribution():
    """Pr(k) ∝ |c_k| → coeffs 3 and 1 give 75%/25%."""
    H = Hamiltonian.time_independent([
        (PauliString("Z"), 3.0),
        (PauliString("X"), 1.0),
    ])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=0)

    counts = [0, 0]
    for _ in range(10000):
        k, _, _ = sampler.sample_pauli_index(0.0)
        counts[k] += 1

    ratio = counts[0] / sum(counts)
    assert abs(ratio - 0.75) < 0.02


def test_pauli_sign_negative():
    H = Hamiltonian.time_independent([(PauliString("Z"), -2.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=42)

    _, _, sign = sampler.sample_pauli_index(0.0)
    assert sign == -1.0


def test_p_delta_formula():
    delta = 0.3
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=delta, total_time=1.0)

    expected = 2.0 / (3.0 - np.cos(delta))
    assert abs(sampler.p_delta - expected) < 1e-10


def test_angle_values():
    """Angle should be ±Δ or π."""
    delta = 0.3
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=delta, total_time=1.0, seed=42)

    for _ in range(200):
        angle, ell = sampler.sample_angle(sign=1.0)
        if ell == 0:
            assert abs(angle - delta) < 1e-12
        else:
            assert abs(angle - np.pi) < 1e-12


def test_angle_sign_delta():
    """Negative sign should flip the Δ angle."""
    delta = 0.3
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=delta, total_time=1.0, seed=1)

    for _ in range(200):
        angle, ell = sampler.sample_angle(sign=-1.0)
        if ell == 0:
            assert abs(angle + delta) < 1e-12
        else:
            assert abs(angle - np.pi) < 1e-12


def test_angle_distribution():
    """Empirical Pr(ℓ=0) should match p_Δ."""
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=0)

    ell_zero = sum(sampler.sample_angle(1.0)[1] == 0 for _ in range(20000))
    empirical = ell_zero / 20000
    assert abs(empirical - sampler.p_delta) < 0.02


from continuous_tepai import SampledCircuit, PauliRotation


# -- full circuit tests --

def test_weight_prefactor_formula():
    delta = 0.3
    T = 2.0
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=delta, total_time=T)

    expected = np.exp(2.0 * 1.0 * T * np.tan(delta / 2.0))
    assert abs(sampler.weight_prefactor - expected) < 1e-10


def test_sample_circuit_returns_type():
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=0)

    circuit = sampler.sample_circuit()
    assert isinstance(circuit, SampledCircuit)
    for r in circuit.rotations:
        assert isinstance(r, PauliRotation)


def test_sample_circuit_weight_magnitude():
    """|g_ω| should equal the weight prefactor (sign product is ±1)."""
    H = Hamiltonian.time_independent([(PauliString("Z"), 1.0)])
    sampler = ContinuousTEPAI(H, delta=0.3, total_time=1.0, seed=0)

    for _ in range(20):
        circuit = sampler.sample_circuit()
        assert abs(abs(circuit.weight) - sampler.weight_prefactor) < 1e-10


def test_sample_circuit_angles_valid():
    """Every rotation angle must be ±Δ or π."""
    delta = 0.4
    H = Hamiltonian.time_independent([
        (PauliString("Z"), 1.0),
        (PauliString("X"), -0.5),
    ])
    sampler = ContinuousTEPAI(H, delta=delta, total_time=1.0, seed=42)

    for _ in range(10):
        circuit = sampler.sample_circuit()
        for r in circuit.rotations:
            assert abs(abs(r.angle) - delta) < 1e-12 or abs(r.angle - np.pi) < 1e-12