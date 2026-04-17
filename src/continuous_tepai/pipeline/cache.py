"""Data path generation and .npz caching.

Folder layout::

    Data/
    └── q{n}_J{j}_td{0|1}_T{T}_dT{dT}_{obs}_{init}/
        ├── trotter{order}_N{N}_depth{depth}[_chi{chi}].npz
        └── tepai_d{d}_S{n_circuits}_tstart{tstart}[_chi{chi}].npz

The folder captures physics parameters shared by all simulations in it
(system size, coupling, observable, etc.).  The filename encodes the
method-specific parameters (Trotter steps, TE-PAI delta exponent, …).

Each ``.npz`` file contains arbitrary numpy arrays (``times``,
``expectation_values``, ``raw_estimates``, …) plus a JSON-encoded
``_metadata`` entry with the full parameter dict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ── top-level data directory ────────────────────────────────────────────

DATA_ROOT = Path("Data")


# ── float formatting for path components ────────────────────────────────

def _fmt(x: float) -> str:
    """Format a float cleanly for use in directory / file names.

    * ``0.1``  → ``"0.1"``
    * ``3.0``  → ``"3.0"``
    * ``0.25`` → ``"0.25"``
    """
    s = f"{x:g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


# ── folder name ─────────────────────────────────────────────────────────

def experiment_folder_name(params: dict[str, Any]) -> str:
    """Build the subfolder name from physics parameters.

    Format: ``q{n}_J{j}_td{0|1}_T{T}_dT{dT}_{observable}_{initial_state}``

    Example: ``q10_J0.1_td0_T0.5_dT0.1_X0_zeros``
    """
    n = params["n_qubits"]
    j = params["j"]
    td = int(bool(params.get("time_dependent", False)))
    T = params["total_time"]
    dT = params["dt"]
    obs = params["observable"]
    init = params["initial_state"]
    return f"q{n}_J{_fmt(j)}_td{td}_T{_fmt(T)}_dT{_fmt(dT)}_{obs}_{init}"


# ── file names ──────────────────────────────────────────────────────────

def trotter_filename(params: dict[str, Any]) -> str:
    """Build the Trotter data filename.

    Format: ``trotter{order}_N{N}_depth{depth}[_chi{chi}].npz``

    *chi* is included only when ``max_bond`` is set (tensor-network run).
    """
    order = params.get("trotter_order", 1)
    N = params["N"]
    depth = params.get("depth", 1)
    chi = params.get("max_bond")

    name = f"trotter{order}_N{N}_depth{depth}"
    if chi is not None:
        name += f"_chi{chi}"
    return name + ".npz"


def tepai_filename(params: dict[str, Any]) -> str:
    r"""Build the TE-PAI data filename.

    Format: ``tepai_d{d}_S{n_circuits}_tstart{tstart}[_chi{chi}].npz``

    *d* is the integer such that Δ = π / 2^d (so ``pi_over_delta = 2^d``).
    *chi* is included only for tensor-network (MPS) runs.
    """
    pod = params["pi_over_delta"]

    # Validate that pi_over_delta is a power of 2.
    d = round(math.log2(pod))
    if 2**d != pod:
        raise ValueError(
            f"pi_over_delta must be a power of 2, got {pod} "
            f"(nearest: 2^{d} = {2**d})"
        )

    n_circuits = params.get("n_circuits", 1)
    tstart = params.get("tepai_start_time", 0.0)
    chi = params.get("max_bond")

    name = f"tepai_d{d}_S{n_circuits}_tstart{_fmt(tstart)}"
    if chi is not None:
        name += f"_chi{chi}"
    return name + ".npz"


# ── full path resolution ────────────────────────────────────────────────

def resolve_data_path(params: dict[str, Any]) -> Path:
    """Return the full relative path for the experiment's cached data file.

    Combines :func:`experiment_folder_name` with the method-specific
    filename (Trotter or TE-PAI).
    """
    folder = experiment_folder_name(params)
    exp_type = params.get("type", "")
    if exp_type == "trotter":
        fname = trotter_filename(params)
    elif exp_type == "tepai":
        fname = tepai_filename(params)
    else:
        raise ValueError(f"Unknown experiment type {exp_type!r}")
    return DATA_ROOT / folder / fname


def is_cached(params: dict[str, Any]) -> bool:
    """Return True if cached data exists on disk for *params*."""
    return resolve_data_path(params).exists()


# ── .npz I/O ────────────────────────────────────────────────────────────

def _metadata_to_array(metadata: dict[str, Any]) -> np.ndarray:
    """Encode *metadata* as a rank-0 numpy string array for ``.npz`` storage."""
    return np.array(json.dumps(metadata, default=str))


def _metadata_from_array(arr: np.ndarray) -> dict[str, Any]:
    """Decode a metadata array written by :func:`_metadata_to_array`."""
    return json.loads(str(arr))


def save_results(
    params: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    **arrays: np.ndarray,
) -> Path:
    """Save experiment results as a ``.npz`` file.

    Parameters
    ----------
    params :
        Merged parameter dict — used to compute the data path.
    metadata :
        Extra metadata to embed inside the file.  When *None*, the full
        *params* dict is used as metadata.
    **arrays :
        Named numpy arrays to persist (``times``, ``expectation_values``, …).

    Returns
    -------
    Path
        The relative path where the file was written.
    """
    path = resolve_data_path(params)
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = params

    np.savez(
        path,
        _metadata=_metadata_to_array(metadata),
        **arrays,
    )
    return path


def load_results(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load cached results from a ``.npz`` file.

    Parameters
    ----------
    params :
        Merged parameter dict — used to compute the data path.

    Returns
    -------
    metadata :
        The metadata dict embedded at save time.
    arrays :
        Dict mapping array names to numpy arrays.

    Raises
    ------
    FileNotFoundError
        If no cached file exists.
    """
    path = resolve_data_path(params)
    if not path.exists():
        raise FileNotFoundError(f"No cached data at {path}")

    data = np.load(path, allow_pickle=True)
    metadata = _metadata_from_array(data["_metadata"])
    arrays = {k: data[k] for k in data.files if k != "_metadata"}
    return metadata, arrays
