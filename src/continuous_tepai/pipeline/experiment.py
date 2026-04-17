"""Experiment dispatch and helper builders.

This module wires config parameters to the ``continuous_tepai`` API:

* :func:`build_hamiltonian` — creates a :class:`Hamiltonian` from config.
* :func:`parse_observable` — turns ``"X0"`` into a :class:`PauliString`.
* :func:`get_backend` — instantiates the right backend class.
* :func:`run_experiment` — main entry point called by ``runner.py``.

Experiment type implementations (``run_trotter``, ``run_tepai``) are stubs
that will be filled in during the numerics phase.  The pipeline framework
(config → cache check → dispatch → save → update config) is fully
functional around them.
"""

from __future__ import annotations

import logging
import math
from itertools import product
from typing import Any

import numpy as np

from ..hamiltonian import Hamiltonian, PauliString

log = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────

def parse_observable(obs_str: str, n_qubits: int) -> PauliString:
    """Parse a short observable string into a :class:`PauliString`.

    Format: ``"{P}{q}"`` where *P* is one of ``X``, ``Y``, ``Z`` and *q*
    is the (zero-based) qubit index.

    Example: ``parse_observable("X0", 4)`` → ``PauliString("IIIX")``

    The codebase uses big-endian labels: ``label[i]`` acts on qubit
    ``n − 1 − i``, so qubit 0 occupies the *rightmost* character.
    """
    if len(obs_str) < 2:
        raise ValueError(f"Observable string too short: {obs_str!r}")
    pauli_char = obs_str[0].upper()
    if pauli_char not in "XYZ":
        raise ValueError(
            f"Observable must start with X, Y, or Z; got {pauli_char!r}"
        )
    qubit = int(obs_str[1:])
    if not 0 <= qubit < n_qubits:
        raise ValueError(
            f"Qubit index {qubit} out of range for {n_qubits}-qubit system"
        )
    label = ["I"] * n_qubits
    label[n_qubits - 1 - qubit] = pauli_char  # big-endian convention
    return PauliString("".join(label))


def normalize_initial_state(name: str) -> str:
    """Normalize initial-state names to what the backends accept.

    ``"zeros"`` → ``"zero"`` ; everything else passed through.
    """
    return "zero" if name == "zeros" else name


def get_backend(params: dict[str, Any]):
    """Instantiate the backend specified by ``params["backend"]``.

    Returns an object satisfying the :class:`Backend` protocol.
    """
    backend_name = params.get("backend", "qulacs")
    max_bond = params.get("max_bond")

    if backend_name == "qulacs":
        from ..backends import QulacsBackend
        return QulacsBackend()

    if backend_name == "qiskit":
        from ..backends import QiskitBackend
        return QiskitBackend()

    if backend_name == "mps":
        from ..backends import MPSBackend
        kw: dict = {}
        if max_bond is not None:
            kw["max_bond"] = int(max_bond)
        return MPSBackend(**kw)

    raise ValueError(f"Unknown backend: {backend_name!r}")


def build_hamiltonian(params: dict[str, Any]) -> Hamiltonian:
    """Build a :class:`Hamiltonian` from config parameters.

    Currently supports ``"spin_chain"`` — the transverse-field Ising /
    Heisenberg chain::

        H(t) = J(t) Σ_{k} (XX + YY + ZZ)_{k,k+1}  +  Σ_k freq_k · Z_k

    where ``J(t)`` is constant (``j``) when ``time_dependent`` is false,
    and ``cos(j · π · t)`` when true.  Site frequencies are drawn from
    ``Uniform(-1, 1)`` with the given ``seed``.
    """
    ham_type = params.get("hamiltonian", "spin_chain")
    n = params["n_qubits"]
    j = params["j"]
    td = params.get("time_dependent", False)
    seed = params.get("seed", 0)

    if ham_type != "spin_chain":
        raise NotImplementedError(
            f"Hamiltonian type {ham_type!r} is not yet implemented. "
            "Currently supported: 'spin_chain'."
        )

    rng = np.random.default_rng(seed)
    freqs = rng.uniform(-1, 1, size=n)

    if td:
        def J(t, _j=j):
            return float(np.cos(_j * np.pi * t))
    else:
        def J(t, _j=j):
            return _j

    terms = [
        (gate, [k, (k + 1) % n], J)
        for k, gate in product(range(n), ["XX", "YY", "ZZ"])
    ]
    terms += [
        ("Z", [k], lambda t, f=float(freqs[k]): f)
        for k in range(n)
    ]
    return Hamiltonian.from_local_terms(n, terms)


# ── experiment type implementations (stubs) ─────────────────────────────

def run_trotter(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run a Trotterized time-evolution experiment.

    **Not yet implemented.**  When filled in this function should:

    1.  Build the Hamiltonian via :func:`build_hamiltonian`.
    2.  Construct Trotter circuits of the requested order / depth.
    3.  Evaluate the observable at each snapshot time ``[0, dT, 2·dT, …, T]``.
    4.  Return ``{"times": …, "expectation_values": …}`` as numpy arrays.

    Parameters
    ----------
    params : dict
        Merged parameter dict.  Relevant keys:

        * ``n_qubits``, ``j``, ``time_dependent``, ``hamiltonian``, ``seed``
        * ``total_time``, ``dt``
        * ``N`` — Trotter step count
        * ``trotter_order`` — 1 (first-order) or 2 (second-order)
        * ``depth`` — 1 (linear step count) or 2 (adaptive quadratic)
        * ``backend``, ``max_bond``
        * ``observable``, ``initial_state``
    """
    raise NotImplementedError(
        "Trotter experiment runner not yet implemented.  "
        "This stub will be filled in during the numerics phase."
    )


def run_tepai(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run a Continuous TE-PAI experiment.

    **Not yet implemented.**  When filled in this function should:

    1.  Build the Hamiltonian via :func:`build_hamiltonian`.
    2.  For each snapshot time in ``[dT, 2·dT, …, T]``:
        a.  Create a :class:`ContinuousTEPAI` sampler with
            ``delta = π / pi_over_delta`` and ``total_time = t``.
        b.  Sample ``n_circuits`` circuits.
        c.  Evaluate each circuit on the chosen backend.
        d.  Compute the weighted mean estimator for ⟨O(t)⟩.
    3.  Prepend the ``t = 0`` value (bare initial-state expectation).
    4.  Return ``{"times": …, "expectation_values": …,
        "raw_estimates": …}`` as numpy arrays.

    Parameters
    ----------
    params : dict
        Merged parameter dict.  Relevant keys:

        * ``n_qubits``, ``j``, ``time_dependent``, ``hamiltonian``, ``seed``
        * ``total_time``, ``dt``
        * ``pi_over_delta`` — integer 2^d giving Δ = π / 2^d
        * ``n_circuits`` — number of sampled circuits per snapshot
        * ``tepai_start_time`` — for future hybrid Trotter+TE-PAI support
        * ``backend``, ``max_bond``
        * ``observable``, ``initial_state``
    """
    raise NotImplementedError(
        "TE-PAI experiment runner not yet implemented.  "
        "This stub will be filled in during the numerics phase."
    )


# ── main dispatch ───────────────────────────────────────────────────────

_RUNNERS = {
    "trotter": run_trotter,
    "tepai": run_tepai,
}


def run_experiment(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Dispatch to the right experiment runner based on ``params["type"]``.

    Parameters
    ----------
    params :
        Merged parameter dict (defaults + experiment overrides).

    Returns
    -------
    dict[str, np.ndarray]
        Named arrays to be persisted by the caching layer.

    Raises
    ------
    NotImplementedError
        If the experiment type's runner is still a stub.
    ValueError
        If ``params["type"]`` is unknown.
    """
    exp_type = params.get("type", "")
    runner = _RUNNERS.get(exp_type)
    if runner is None:
        raise ValueError(
            f"Unknown experiment type {exp_type!r}.  "
            f"Available: {list(_RUNNERS.keys())}"
        )
    return runner(params)
