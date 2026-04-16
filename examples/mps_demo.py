"""Minimal Continuous TE-PAI example using the MPS backend.

Compares the MPS estimator against Qulacs on a small spin chain.

Run:
    uv run python examples/mps_demo.py
"""

from __future__ import annotations

import numpy as np

from continuous_tepai import ContinuousTEPAI, PauliString
from continuous_tepai.backends import MPSBackend

from spin_chain import spin_chain_hamiltonian


def main() -> None:
    n_qubits = 4
    T = 0.4
    delta = 0.3
    n_circuits = 32
    seed = 0

    rng = np.random.default_rng(seed)
    freqs = rng.uniform(-1, 1, size=n_qubits)

    ham = spin_chain_hamiltonian(n_qubits, freqs, coef=0.0)
    sampler = ContinuousTEPAI(ham, delta=delta, total_time=T, seed=seed)

    # Observable: Z on the leftmost qubit.
    obs = PauliString("Z" + "I" * (n_qubits - 1))
    mps = MPSBackend()  # exact (no bond-dim truncation) by default

    circuits = sampler.sample_circuits(n_circuits)
    estimates = [
        c.weight * mps.expectation(c.rotations, obs, n_qubits, initial_state="plus")
        for c in circuits
    ]
    mean = float(np.mean(estimates))
    sem = float(np.std(estimates, ddof=1) / np.sqrt(n_circuits))

    print(f"n_qubits={n_qubits}, T={T}, Δ={delta}, circuits={n_circuits}")
    print(f"⟨Z⟩ estimate (MPS backend) = {mean:+.4f} ± {sem:.4f}")


if __name__ == "__main__":
    main()
