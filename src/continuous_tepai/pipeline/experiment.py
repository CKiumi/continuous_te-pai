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


def _mp_eval_circuit_with_bond(args: tuple) -> tuple[float, int]:
    """Pool worker: evaluate one circuit and also return MPS max-bond."""
    rotations, obs, n, init = args
    return _worker_backend.expectation_with_bond(
        rotations, obs, n, initial_state=init,
    )


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

    # Terminal overhead (at t = T) — upper bound for the whole run.
    terminal_sampler = ContinuousTEPAI(ham, delta=delta, total_time=T, seed=seed)
    terminal_overhead = terminal_sampler.weight_prefactor

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

    log.info(
        "    TE-PAI overhead at T=%.3f: exp(2‖c‖₁_avg·T·tan(Δ/2)) = %.4g",
        T, terminal_overhead,
    )

    # t = 0 measurement (no evolution)
    if use_mp:
        ev0 = pool.apply(_mp_eval_circuit, args=(([], obs, n, init),))
    else:
        ev0 = backend.expectation([], obs, n, initial_state=init)
    means = [ev0]
    stds = [0.0]
    overheads = [1.0]  # t = 0: no evolution, overhead exp(0) = 1.

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
            overheads.append(float(sampler.weight_prefactor))

            log.info(
                "      snap %d/%d (T=%.2f): ⟨O⟩ = %.4f ± %.4f, overhead = %.4g",
                i, n_snapshots, t_snap, means[-1], stds[-1], overheads[-1],
            )
    finally:
        if use_mp:
            pool.close()
            pool.join()

    return {
        "times": snap_times,
        "expectation_values": np.array(means),
        "std_errors": np.array(stds),
        "overheads": np.array(overheads),
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
            "overheads": tepai_data["overheads"],
        }
    else:
        log.info("    [snapshot] Running TE-PAI estimate …")
        tepai_out = run_tepai(params)
        save_results(
            tepai_params,
            columns=["times", "expectation_values", "std_errors", "overheads"],
            arrays=[
                tepai_out["times"],
                tepai_out["expectation_values"],
                tepai_out["std_errors"],
                tepai_out["overheads"],
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


def run_bond_tracking(params: dict[str, Any]) -> dict[str, np.ndarray]:
    r"""Track MPS bond dimension over time for both Trotter and TE-PAI.

    Runs a Trotter reference and a Continuous TE-PAI estimate on the MPS
    backend with a large (but finite) bond-dimension cap, and records the
    quimb ``psi.max_bond()`` at every snapshot for both methods.  At each
    snapshot the TE-PAI bond value is the maximum across all sampled
    circuits at that time.

    Produces a 2×1 subplot PDF — expectation values on top, MPS max-bond
    on the bottom.

    Only valid with ``backend = "mps"``.

    Returns
    -------
    dict with ``times``, ``trotter_values``, ``trotter_bonds``,
    ``tepai_mean``, ``tepai_std``, ``tepai_max_bond``, ``overheads``.
    """
    from .trotter import build_trotter_rotations, execute_trotter_mps_with_bonds
    from .cache import (
        DATA_ROOT, experiment_folder_name, resolve_data_path,
        save_csv, save_results, load_results,
    )
    from .plotting import bond_tracking_plot
    from ..te_pai import ContinuousTEPAI

    if params.get("backend", "mps") != "mps":
        raise ValueError(
            "bond_tracking experiment requires backend='mps' "
            f"(got {params.get('backend')!r})."
        )

    ham = build_hamiltonian(params)
    n = params["n_qubits"]
    T = params["total_time"]
    dt = params["dt"]
    N = params["N"]
    depth = params.get("depth", 1)
    pod = params["pi_over_delta"]
    delta = float(np.pi / pod)
    n_circuits = params.get("n_circuits", 32)
    seed = params.get("seed", 0)
    init = normalize_initial_state(params["initial_state"])
    obs = parse_observable(params["observable"], n)
    chi = params.get("max_bond")

    n_snapshots = round(T / dt)
    snap_times = np.linspace(0, T, n_snapshots + 1)

    # ── short-circuit cache: reload the full CSV if it exists ─────
    cache_path = resolve_data_path(params)
    sidecar_path = cache_path.with_name(
        cache_path.stem + "_finalbonds.csv"
    )
    if cache_path.exists() and sidecar_path.exists():
        log.info("    [bond_tracking] Loading cached CSV …")
        meta, data = load_results(params)
        result = {
            "times": data["times"],
            "trotter_values": data["trotter_values"],
            "trotter_bonds": data["trotter_bonds"].astype(int),
            "tepai_mean": data["tepai_mean"],
            "tepai_std": data["tepai_std"],
            "tepai_max_bond": data["tepai_max_bond"].astype(int),
            "overheads": data["overheads"],
        }
        trotter_gate_count = int(float(meta.get("trotter_gate_count", "0")))
        tepai_avg_gate_count = float(meta.get("tepai_avg_gate_count", "nan"))
        from .cache import load_csv
        _, side = load_csv(sidecar_path)
        tepai_final_bonds = side["max_bond"].astype(int)
    else:
        log.info(
            "    BondTrack: n=%d, χ=%s, N=%d, depth=%d, Δ=π/%d, N_s=%d, n_snap=%d",
            n, chi, N, depth, pod, n_circuits, n_snapshots,
        )

        # ── Trotter with bond tracking ────────────────────────────
        log.info("    [bond_tracking] Trotter reference …")
        rots = build_trotter_rotations(ham, T, N, n_snapshots, depth=depth)
        trotter_gate_count = int(sum(len(r) for r in rots))
        trotter_vals, trotter_bonds = execute_trotter_mps_with_bonds(
            rots, obs, n, init, max_bond=chi,
        )
        log.info(
            "    [bond_tracking] Trotter done; max bond across time = %d, "
            "total gate count = %d",
            int(trotter_bonds.max()), trotter_gate_count,
        )

        # ── TE-PAI with bond tracking ─────────────────────────────
        n_workers = _worker_count()
        log.info(
            "    [bond_tracking] TE-PAI: Δ=π/%d, N_s=%d, workers=%d",
            pod, n_circuits, n_workers,
        )
        ctx = mp.get_context("spawn" if platform.system() == "Darwin" else "fork")
        pool = ctx.Pool(
            n_workers,
            initializer=_mp_init_backend,
            initargs=(params,),
        )

        # t = 0 — no evolution
        ev0, bond0 = pool.apply(
            _mp_eval_circuit_with_bond, args=(([], obs, n, init),),
        )
        tepai_means = [float(ev0)]
        tepai_stds = [0.0]
        tepai_max_bonds = [int(bond0)]
        overheads = [1.0]
        tepai_gate_counts_final: list[int] = []
        tepai_final_bonds_list: list[int] = []

        try:
            for i, t_snap in enumerate(snap_times[1:], start=1):
                sampler = ContinuousTEPAI(
                    ham, delta=delta, total_time=t_snap, seed=seed + i,
                )
                circuits = sampler.sample_circuits(n_circuits)
                tasks = [(c.rotations, obs, n, init) for c in circuits]
                raw = pool.map(_mp_eval_circuit_with_bond, tasks)
                evs = np.array([v for v, _ in raw])
                bonds = np.array([b for _, b in raw], dtype=int)
                values = np.array([c.weight * v for c, v in zip(circuits, evs)])

                tepai_means.append(float(np.mean(values)))
                tepai_stds.append(float(np.std(values, ddof=1) / np.sqrt(n_circuits)))
                tepai_max_bonds.append(int(bonds.max()))
                overheads.append(float(sampler.weight_prefactor))
                if i == n_snapshots:
                    tepai_gate_counts_final = [c.gate_count for c in circuits]
                    tepai_final_bonds_list = [int(b) for b in bonds]

                log.info(
                    "      snap %d/%d (T=%.2f): ⟨O⟩ = %.4f ± %.4f, "
                    "max χ over %d circuits = %d",
                    i, n_snapshots, t_snap,
                    tepai_means[-1], tepai_stds[-1],
                    n_circuits, tepai_max_bonds[-1],
                )
        finally:
            pool.close()
            pool.join()

        tepai_avg_gate_count = (
            float(np.mean(tepai_gate_counts_final))
            if tepai_gate_counts_final else float("nan")
        )
        tepai_final_bonds = np.array(tepai_final_bonds_list, dtype=int)

        result = {
            "times": snap_times,
            "trotter_values": trotter_vals,
            "trotter_bonds": np.asarray(trotter_bonds, dtype=int),
            "tepai_mean": np.array(tepai_means),
            "tepai_std": np.array(tepai_stds),
            "tepai_max_bond": np.array(tepai_max_bonds, dtype=int),
            "overheads": np.array(overheads),
        }

        # ── save CSV ──────────────────────────────────────────────
        metadata = {
            **params,
            "trotter_gate_count": trotter_gate_count,
            "tepai_avg_gate_count": tepai_avg_gate_count,
        }
        save_results(
            params,
            columns=[
                "times", "trotter_values", "trotter_bonds",
                "tepai_mean", "tepai_std", "tepai_max_bond", "overheads",
            ],
            arrays=[
                result["times"],
                result["trotter_values"],
                result["trotter_bonds"],
                result["tepai_mean"],
                result["tepai_std"],
                result["tepai_max_bond"],
                result["overheads"],
            ],
            metadata=metadata,
        )
        log.info("    [bond_tracking] Data saved to %s", cache_path)

        # Sidecar: per-circuit max-bond at the final snapshot (for the histogram).
        save_csv(
            sidecar_path,
            metadata={"snapshot_time": T, "n_circuits": n_circuits},
            columns=["max_bond"],
            arrays=[tepai_final_bonds],
        )
        log.info("    [bond_tracking] Final-bond histogram saved to %s", sidecar_path)

    # ── PDF plot (always regenerated) ─────────────────────────────
    exp_name = params.get("name", "bond_tracking")
    folder = DATA_ROOT / experiment_folder_name(params)
    folder.mkdir(parents=True, exist_ok=True)
    plot_path = folder / f"{exp_name}.pdf"

    bond_tracking_plot(
        result["times"],
        result["trotter_values"],
        result["tepai_mean"],
        result["tepai_std"],
        result["trotter_bonds"],
        result["tepai_max_bond"],
        n_qubits=n,
        observable_str=params["observable"],
        n_circuits=n_circuits,
        max_bond_cap=chi,
        trotter_gate_count=trotter_gate_count,
        tepai_avg_gate_count=tepai_avg_gate_count,
        trotter_final_bond=int(result["trotter_bonds"][-1]),
        tepai_final_bonds=tepai_final_bonds,
        output_path=plot_path,
    )
    log.info("    [bond_tracking] Plot saved to %s", plot_path)

    return result


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
    log.info(
        "    Circuits overhead: exp(2‖c‖₁_avg·T·tan(Δ/2)) = %.4g "
        "(expected gate count Λ = %.2f)",
        sampler.weight_prefactor, sampler.expected_gate_count,
    )
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


def run_overhead(params: dict[str, Any]) -> dict[str, np.ndarray]:
    r"""Compute the TE-PAI overhead as a function of time for several qubit counts.

    Overhead:
        :math:`\exp(2\,\|c\|_{1,\text{avg}}(T)\,T\,\tan(\Delta/2))`

    Required config keys beyond the usual ones:
    * ``qubit_list`` — list of qubit counts to sweep.

    Writes a CSV (one column per qubit count) and a PDF plot into
    ``Data/overhead_.../{name}_d{d}.{csv,pdf}``.
    """
    from .cache import (
        DATA_ROOT, overhead_folder_name, overhead_filename, save_csv,
    )
    from .plotting import overhead_plot

    qubit_list = params.get("qubit_list")
    if not qubit_list:
        raise ValueError(
            "Overhead experiments require 'qubit_list' — a list of qubit counts."
        )
    qubit_list = [int(n) for n in qubit_list]

    T = params["total_time"]
    dt = params["dt"]
    pod = params["pi_over_delta"]
    delta = float(np.pi / pod)
    tan_half = float(np.tan(delta / 2.0))

    n_snapshots = round(T / dt)
    times = np.linspace(0, T, n_snapshots + 1)

    log.info(
        "    Overhead: qubits=%s, Δ=π/%d, n_snap=%d",
        qubit_list, pod, n_snapshots,
    )

    overheads_by_n: dict[int, np.ndarray] = {}
    for n in qubit_list:
        ham = build_hamiltonian({**params, "n_qubits": n})
        oh = np.empty_like(times)
        oh[0] = 1.0
        for i, t in enumerate(times[1:], start=1):
            c1_avg = ham.l1_norm_avg(float(t))
            oh[i] = float(np.exp(2.0 * c1_avg * float(t) * tan_half))
        overheads_by_n[n] = oh
        log.info("      n=%d: overhead(T=%.3f) = %.4g", n, T, oh[-1])

    # ── save CSV + PDF ────────────────────────────────────────────
    folder = DATA_ROOT / overhead_folder_name(params)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / overhead_filename(params)
    pdf_path = folder / (overhead_filename(params).removesuffix(".csv") + ".pdf")

    columns = ["times"] + [f"q{n}" for n in qubit_list]
    arrays = [times] + [overheads_by_n[n] for n in qubit_list]
    metadata = {
        "type": "overhead",
        "hamiltonian": params.get("hamiltonian", "spin_chain"),
        "qubit_list": ",".join(str(n) for n in qubit_list),
        "j": params["j"],
        "time_dependent": bool(params.get("time_dependent", False)),
        "total_time": T,
        "dt": dt,
        "pi_over_delta": pod,
        "delta": delta,
        "seed": params.get("seed", 0),
    }
    save_csv(csv_path, metadata, columns, arrays)
    log.info("    [overhead] Data saved to %s", csv_path)

    overhead_plot(
        times,
        overheads_by_n,
        pi_over_delta=pod,
        j=float(params["j"]),
        time_dependent=bool(params.get("time_dependent", False)),
        output_path=pdf_path,
    )
    log.info("    [overhead] Plot saved to %s", pdf_path)

    return {
        "times": times,
        **{f"q{n}": overheads_by_n[n] for n in qubit_list},
    }


# ── main dispatch ───────────────────────────────────────────────────────

_RUNNERS = {
    "trotter": run_trotter,
    "tepai": run_tepai,
    "snapshot": run_snapshot,
    "circuits": run_circuits,
    "overhead": run_overhead,
    "bond_tracking": run_bond_tracking,
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
