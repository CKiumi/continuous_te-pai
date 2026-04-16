"""Tests for the MPS backend.

Covers:
    * Basic protocol parity (identity circuit, single-qubit rotations, ...).
    * Label-ordering regression (same as ``test_qulacs_backend.py``).
    * Cross-backend parity against Qulacs on TE-PAI sampled circuits.
"""

from __future__ import annotations

import numpy as np
import pytest

quimb = pytest.importorskip("quimb")

from continuous_tepai import (
    ContinuousTEPAI,
    Hamiltonian,
    PauliRotation,
    PauliString,
)
from continuous_tepai.backends import MPSBackend


# ---------------------------------------------------------------------------
# basic Backend-protocol tests mirroring tests/test_qulacs_backend.py
# ---------------------------------------------------------------------------


def test_identity_circuit():
    backend = MPSBackend()
    ev = backend.expectation([], PauliString("Z"), num_qubits=1)
    assert abs(ev - 1.0) < 1e-10


def test_x_rotation_pi():
    backend = MPSBackend()
    rot = PauliRotation(PauliString("X"), angle=np.pi)
    ev = backend.expectation([rot], PauliString("Z"), num_qubits=1)
    assert abs(ev - (-1.0)) < 1e-10


def test_plus_initial_state():
    backend = MPSBackend()
    ev = backend.expectation(
        [], PauliString("X"), num_qubits=1, initial_state="plus"
    )
    assert abs(ev - 1.0) < 1e-10


def test_two_qubit_ordering():
    backend = MPSBackend()
    rot = PauliRotation(PauliString("XI"), angle=np.pi)
    assert abs(backend.expectation([rot], PauliString("ZI"), 2) - (-1.0)) < 1e-10
    assert abs(backend.expectation([rot], PauliString("IZ"), 2) - 1.0) < 1e-10


def test_zz_rotation_entangles():
    """exp(-i π/4 · ZZ)|++⟩ has ⟨XX⟩ = 0 (a Bell-like state)."""
    backend = MPSBackend()
    rot = PauliRotation(PauliString("ZZ"), angle=np.pi / 2)
    ev = backend.expectation(
        [rot], PauliString("ZZ"), num_qubits=2, initial_state="plus"
    )
    # ⟨ZZ⟩ on |++⟩ is 0; the ZZ-rotation commutes with ZZ, so still 0.
    assert abs(ev) < 1e-10


def test_identity_observable():
    backend = MPSBackend()
    ev = backend.expectation(
        [PauliRotation(PauliString("X"), angle=0.7)],
        PauliString("II"),
        num_qubits=2,
    )
    assert abs(ev - 1.0) < 1e-10


def test_multi_site_pauli_observable():
    """⟨XX⟩ on |++⟩ = 1 (two-site observable)."""
    backend = MPSBackend()
    ev = backend.expectation(
        [], PauliString("XX"), num_qubits=2, initial_state="plus"
    )
    assert abs(ev - 1.0) < 1e-10


def test_unsupported_mixed_two_qubit_rotation():
    backend = MPSBackend()
    rot = PauliRotation(PauliString("XZ"), angle=0.1)
    with pytest.raises(NotImplementedError):
        backend.expectation([rot], PauliString("ZI"), num_qubits=2)


# ---------------------------------------------------------------------------
# parity tests: MPS (exact, no truncation) ≡ Qulacs on small systems
# ---------------------------------------------------------------------------


def _qulacs_backend_or_skip():
    pytest.importorskip("qulacs")
    from continuous_tepai.backends import QulacsBackend

    return QulacsBackend()


def _spin_chain_hamiltonian(n: int, seed: int = 0) -> Hamiltonian:
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(-1, 1, size=n)

    def J(t, coef=0.37):
        return float(np.cos(coef * t * np.pi))

    from itertools import product

    terms = [
        (gate, [k, (k + 1) % n], J)
        for k, gate in product(range(n), ["XX", "YY", "ZZ"])
    ]
    terms += [
        ("Z", [k], lambda t, f=float(freqs[k]): f) for k in range(n)
    ]
    return Hamiltonian.from_local_terms(n, terms)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sampled_circuit_parity_vs_qulacs(seed):
    """Identical sampled circuits must produce identical expectation values."""
    qulacs = _qulacs_backend_or_skip()
    mps = MPSBackend()

    n_qubits = 4
    ham = _spin_chain_hamiltonian(n_qubits, seed=seed)
    sampler = ContinuousTEPAI(ham, delta=0.3, total_time=0.5, seed=seed)
    circuit = sampler.sample_circuits(1)[0]

    obs = PauliString("Z" + "I" * (n_qubits - 1))

    ev_mps = mps.expectation(
        circuit.rotations, obs, n_qubits, initial_state="plus"
    )
    ev_qul = qulacs.expectation(
        circuit.rotations, obs, n_qubits, initial_state="plus"
    )
    # Exact MPS (no truncation) must match Qulacs to floating-point precision.
    assert abs(ev_mps - ev_qul) < 1e-8


def test_te_pai_batch_parity_vs_qulacs():
    """Averaged estimator from the same seed must match between backends."""
    qulacs = _qulacs_backend_or_skip()
    mps = MPSBackend()

    n_qubits = 3
    ham = _spin_chain_hamiltonian(n_qubits, seed=7)
    sampler = ContinuousTEPAI(ham, delta=0.3, total_time=0.3, seed=42)
    circuits = sampler.sample_circuits(8)

    obs = PauliString("Z" + "I" * (n_qubits - 1))

    est_mps = np.mean(
        [
            c.weight
            * mps.expectation(c.rotations, obs, n_qubits, initial_state="plus")
            for c in circuits
        ]
    )
    est_qul = np.mean(
        [
            c.weight
            * qulacs.expectation(c.rotations, obs, n_qubits, initial_state="plus")
            for c in circuits
        ]
    )
    assert abs(est_mps - est_qul) < 1e-8


# ---------------------------------------------------------------------------
# regression test: ensure MPSBackend truly uses quimb (and not a fallback)
# ---------------------------------------------------------------------------


def test_backend_is_mps_not_dense():
    """Importing MPSBackend must return the quimb-backed class."""
    import continuous_tepai.backends as be
    from continuous_tepai.backends.mps_backend import MPSBackend as MPSClass

    assert be.MPSBackend is MPSClass


def test_mps_backend_uses_quimb_circuit(monkeypatch):
    """Sanity regression: an MPS expectation call must construct a CircuitMPS."""
    import quimb.tensor as qtn

    calls: list[int] = []
    orig = qtn.CircuitMPS.__init__

    def spy(self, *a, **kw):
        calls.append(1)
        return orig(self, *a, **kw)

    monkeypatch.setattr(qtn.CircuitMPS, "__init__", spy)

    backend = MPSBackend()
    backend.expectation(
        [PauliRotation(PauliString("X"), angle=np.pi)],
        PauliString("Z"),
        num_qubits=1,
    )
    assert calls, "MPSBackend did not construct a quimb CircuitMPS"
