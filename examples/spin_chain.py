"""Spin-chain Hamiltonian used in TE-PAI benchmarks."""

from itertools import product

import numpy as np

from continuous_tepai import Hamiltonian


def spin_chain_hamiltonian(n: int, freqs, coef: float) -> Hamiltonian:
    """H(t) = Σ_{k} cos(coef·π·t) (XX + YY + ZZ)_{k,k+1} + Σ_k freqs[k] Z_k."""

    def J(t):
        return np.cos(coef * t * np.pi)

    terms = [
        (gate, [k, (k + 1) % n], J)
        for k, gate in product(range(n), ["XX", "YY", "ZZ"])
    ]
    terms += [("Z", [k], lambda t, k=k: float(freqs[k])) for k in range(n)]
    return Hamiltonian.from_local_terms(n, terms)