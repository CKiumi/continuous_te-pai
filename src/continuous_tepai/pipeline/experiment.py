"""Experiment dispatch and helper builders.

This module wires config parameters to the ``continuous_tepai`` API:

* :func:`build_hamiltonian` — creates a :class:`Hamiltonian` from config.
* :func:`parse_observable` — turns ``"X0"`` into a :class:`PauliString`.
* :func:`get_backend` — instantiates the right backend class.
* :func:`run_experiment` — main entry point called by ``runner.py``.

Three experiment types are supported:

``"trotter"``
    First-order Trotterized time evolution with snapshot measurements.

``"tepai"``
    Continuous TE-PAI estimation at each snapshot time.

``"snapshot"``
    Combined Trotter + TE-PAI run with a comparison PDF plot, replicating
    the workflow of ``examples/snapshot.ipynb``.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import platform
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ..hamiltonian import Hamiltonian, PauliString
from ..circuit import PauliRotation

log = logging.getLogger(__name__)

# ── multiprocessing ──────────────────────────────────────────────────────

# Process-local backend instance set by _mp_init_backend.
_worker_backend = None


def _worker_count() -> int:
    """10 workers on macOS; all available CPUs on Linux."""
    if platform.system() == "Darwin":
        return 10
    return os.cpu_count() or 1


def _mp_init_backend(backend_params: dict) -> None:
    """Pool initializer: create one backend per worker process."""
    global _worker_backend
    _worker_backend = get_backend(backend_params)


def _mp_eval_circuit(args: tuple) -> float:
    """Pool worker: evaluate one circuit on the process-local backend."""
    rotations, obs, n, init = args
    return _worker_backend.expectation(rotations, obs, n, initial_state=init)


# ── helpers ─────────────────────────────────────────────────────────────

def parse_observable(obs_str: str, n_qubits: int) -> PauliString:
    """Parse a short observable string into a :class:`PauliString`.

    Format: ``"{P}{q}"`` where *P* is one of ``X``, ``Y``, ``Z`` and *q*
    is the (zero-based) qubit index.

    Example: ``parse_observable("X0", 4)`` → ``PauliString("XIII")``

    Matches :meth:`Hamiltonian.from_local_terms`, which places the qubit
    at index ``q`` in the label at position ``label[q]``.  All three
    backends (Qulacs, Qiskit, MPS) then map ``label[i]`` to physical
    qubit ``n - 1 - i`` internally, so the observable and Hamiltonian
    stay on the same physical qubit regardless of backend.
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
    label[qubit] = pauli_char  # match from_local_terms convention
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
    r"""Build a :class:`Hamiltonian` from config parameters.

    Currently supports ``"spin_chain"`` — the Heisenberg-like chain::

        H(t) = J(t) Σ_{k} (XX + YY + ZZ)_{k,k+1}  +  Σ_k freq_k · Z_k

    where ``J(t)`` is constant (``j``) when ``time_dependent`` is false,
    and ``cos(j · π · t)`` when true.

    Site frequencies are drawn from ``Uniform(-1, 1)`` with the given
    ``seed``, unless ``freqs_csv`` points to a CSV file of pre-defined
    frequencies (first ``n_qubits`` rows are used).
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

    # Frequencies: from CSV or random
    freqs_csv = params.get("freqs_csv")
    if freqs_csv is not None:
        all_freqs = np.loadtxt(freqs_csv, delimiter=",")
        freqs = all_freqs[:n].astype(float)
    else:
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


# ── experiment runners ──────────────────────────────────────────────────

def run_trotter(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run a Trotterized time-evolution experiment.

    Constructs Trotter rotations, executes them on the chosen backend,
    and returns snapshot measurements.

    Returns ``{"times": …, "expectation_values": …}``.
    """
    from .trotter import (
        build_trotter_rotations,
        execute_trotter_mps,
        execute_trotter_generic,
    )

    ham = build_hamiltonian(params)
    n = params["n_qubits"]
    T = params["total_time"]
    dt = params["dt"]
    N = params["N"]
    depth = params.get("depth", 1)
    init = normalize_initial_state(params["initial_state"])
    obs = parse_observable(params["observable"], n)
    backend_name = params.get("backend", "qulacs")

    n_snapshots = round(T / dt)
    snap_times = np.linspace(0, T, n_snapshots + 1)

    log.info(
        "    Trotter: N=%d, depth=%d, n_snap=%d, backend=%s",
        N, depth, n_snapshots, backend_name,
    )

    rots = build_trotter_rotations(ham, T, N, n_snapshots, depth=depth)

    if backend_name == "mps":
        evs = execute_trotter_mps(
            rots, obs, n, init,
            max_bond=params.get("max_bond"),
        )
    else:
        backend = get_backend(params)
        evs = execute_trotter_generic(rots, obs, n, init, backend)

    return {"times": snap_times, "expectation_values": evs}


def run_tepai(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run a Continuous TE-PAI experiment.

    For each snapshot time, creates a :class:`ContinuousTEPAI` sampler,
    draws ``n_circuits`` circuits, evaluates weighted expectations,
    and returns the mean estimator with standard errors.

    Returns ``{"times": …, "expectation_values": …, "std_errors": …}``.
    """
    from ..te_pai import ContinuousTEPAI

    ham = build_hamiltonian(params)
    n = params["n_qubits"]
    T = params["total_time"]
    dt = params["dt"]
    pod = params["pi_over_delta"]
    delta = np.pi / pod
    n_circuits = params.get("n_circuits", 32)
    seed = params.get("seed", 0)
    init = normalize_initial_state(params["initial_state"])
    obs = parse_observable(params["observable"], n)
    backend_name = params.get("backend", "qulacs")
    use_mp = backend_name == "mps"

    n_snapshots = round(T / dt)
    snap_times = np.linspace(0, T, n_snapshots + 1)

    if use_mp:
        n_workers = _worker_count()
        log.info(
            "    TE-PAI: Δ=π/%d, N_s=%d, n_snap=%d, backend=%s, workers=%d",
            pod, n_circuits, n_snapshots, backend_name, n_workers,
        )
        ctx = mp.get_context("spawn" if platform.system() == "Darwin" else "fork")
        pool = ctx.Pool(
            n_workers,
            initializer=_mp_init_backend,
            initargs=(params,),
        )
    else:
        log.info(
            "    TE-PAI: Δ=π/%d, N_s=%d, n_snap=%d, backend=%s",
            pod, n_circuits, n_snapshots, backend_name,
        )
        backend = get_backend(params)

    # t = 0 measurement (no evolution)
    if use_mp:
        ev0 = pool.apply(_mp_eval_circuit, args=(([], obs, n, init),))
    else:
        ev0 = backend.expectation([], obs, n, initial_state=init)
    means = [ev0]
    stds = [0.0]

    try:
        for i, t_snap in enumerate(snap_times[1:], start=1):
            sampler = ContinuousTEPAI(
                ham, delta=delta, total_time=t_snap, seed=seed + i,
            )
            circuits = sampler.sample_circuits(n_circuits)

            if use_mp:
                tasks = [(c.rotations, obs, n, init) for c in circuits]
                raw = pool.map(_mp_eval_circuit, tasks)
                values = np.array([c.weight * v for c, v in zip(circuits, raw)])
            else:
                values = np.array([
                    c.weight * backend.expectation(
                        c.rotations, obs, n, initial_state=init,
                    )
                    for c in circuits
                ])

            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1) / np.sqrt(n_circuits)))

            log.info(
                "      snap %d/%d (T=%.2f): ⟨O⟩ = %.4f ± %.4f",
                i, n_snapshots, t_snap, means[-1], stds[-1],
            )
    finally:
        if use_mp:
            pool.close()
            pool.join()

    return {
        "times": snap_times,
        "expectation_values": np.array(means),
        "std_errors": np.array(stds),
    }


def run_snapshot(params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run a combined Trotter + TE-PAI experiment and produce a PDF plot.

    This replicates the workflow of ``examples/snapshot.ipynb``:

    1. Trotter evolution as the reference curve.
    2. Continuous TE-PAI as the estimate with error bars.
    3. Comparison plot saved as PDF in the data folder.

    Each component is cached independently: if only Trotter or only
    TE-PAI parameters change, only the affected component is recomputed.

    Returns ``{"times": …, "trotter_values": …, "tepai_mean": …,
    "tepai_std": …}``.
    """
    from .cache import (
        experiment_folder_name, DATA_ROOT,
        resolve_data_path, is_cached, save_results, load_results,
    )
    from .plotting import snapshot_comparison_plot

    trotter_params = {**params, "type": "trotter"}
    tepai_params = {**params, "type": "tepai"}

    # ── Trotter: load from cache or run + save ────────────────────
    if resolve_data_path(trotter_params).exists():
        log.info("    [snapshot] Loading cached Trotter data …")
        _, trotter_data = load_results(trotter_params)
        trotter_out = {
            "times": trotter_data["times"],
            "expectation_values": trotter_data["expectation_values"],
        }
    else:
        log.info("    [snapshot] Running Trotter reference …")
        trotter_out = run_trotter(params)
        save_results(
            trotter_params,
            columns=["times", "expectation_values"],
            arrays=[trotter_out["times"], trotter_out["expectation_values"]],
        )
        log.info("    [snapshot] Trotter data saved.")

    # ── TE-PAI: load from cache or run + save ─────────────────────
    if resolve_data_path(tepai_params).exists():
        log.info("    [snapshot] Loading cached TE-PAI data …")
        _, tepai_data = load_results(tepai_params)
        tepai_out = {
            "times": tepai_data["times"],
            "expectation_values": tepai_data["expectation_values"],
            "std_errors": tepai_data["std_errors"],
        }
    else:
        log.info("    [snapshot] Running TE-PAI estimate …")
        tepai_out = run_tepai(params)
        save_results(
            tepai_params,
            columns=["times", "expectation_values", "std_errors"],
            arrays=[
                tepai_out["times"],
                tepai_out["expectation_values"],
                tepai_out["std_errors"],
            ],
        )
        log.info("    [snapshot] TE-PAI data saved.")

    times = trotter_out["times"]
    trotter_vals = trotter_out["expectation_values"]
    tepai_mean = tepai_out["expectation_values"]
    tepai_std = tepai_out["std_errors"]

    # ── generate PDF plot ─────────────────────────────────────────
    exp_name = params.get("name", "snapshot")
    folder = DATA_ROOT / experiment_folder_name(params)
    folder.mkdir(parents=True, exist_ok=True)
    plot_path = folder / f"{exp_name}.pdf"

    snapshot_comparison_plot(
        times,
        trotter_vals,
        tepai_mean,
        tepai_std,
        n_qubits=params["n_qubits"],
        observable_str=params["observable"],
        n_circuits=params.get("n_circuits"),
        output_path=plot_path,
    )
    log.info("    [snapshot] Plot saved to %s", plot_path)

    return {
        "times": times,
        "trotter_values": trotter_vals,
        "tepai_mean": tepai_mean,
        "tepai_std": tepai_std,
    }


def run_circuits(params: dict[str, Any]) -> dict[str, Any]:
    """Sample TE-PAI circuits at ``total_time`` and save them to CSV.

    No backend execution; this is the "generate and dump" path used to
    produce stratified-sampling inputs.  The output CSV lives in the
    standard experiment data folder and is written via
    :func:`circuit_io.save_circuits_csv`.
    """
    from ..te_pai import ContinuousTEPAI
    from .cache import resolve_data_path
    from .circuit_io import save_circuits_csv

    ham = build_hamiltonian(params)
    T = params["total_time"]
    pod = params["pi_over_delta"]
    delta = float(np.pi / pod)
    n_circuits = params.get("n_circuits", 10)
    seed = params.get("seed", 0)

    log.info(
        "    Circuits: n_qubits=%d, T=%.3f, Δ=π/%d, n_circuits=%d",
        params["n_qubits"], T, pod, n_circuits,
    )

    sampler = ContinuousTEPAI(ham, delta=delta, total_time=T, seed=seed)
    circuits = sampler.sample_circuits(n_circuits)

    metadata = {
        "type": "circuits",
        "hamiltonian": params.get("hamiltonian", "spin_chain"),
        "n_qubits": params["n_qubits"],
        "total_time": T,
        "pi_over_delta": pod,
        "delta": delta,
        "n_circuits": n_circuits,
        "seed": seed,
        "j": params["j"],
        "time_dependent": bool(params.get("time_dependent", False)),
        "initial_state": params.get("initial_state", ""),
        "weight_prefactor": sampler.weight_prefactor,
        "expected_gate_count": sampler.expected_gate_count,
    }

    path = resolve_data_path(params)
    save_circuits_csv(path, circuits, metadata)

    gate_counts = np.array([c.gate_count for c in circuits], dtype=int)
    log.info(
        "    [circuits] Saved %d circuits to %s (mean length %.1f)",
        n_circuits, path, gate_counts.mean() if n_circuits else 0.0,
    )

    return {"circuits_path": str(path), "gate_counts": gate_counts}


# ── main dispatch ───────────────────────────────────────────────────────

_RUNNERS = {
    "trotter": run_trotter,
    "tepai": run_tepai,
    "snapshot": run_snapshot,
    "circuits": run_circuits,
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
