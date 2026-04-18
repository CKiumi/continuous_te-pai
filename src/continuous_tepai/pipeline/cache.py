"""Data path generation and CSV caching.

Folder layout::

    Data/
    └── q{n}_J{j}_td{0|1}_T{T}_dT{dT}_{obs}_{init}/
        ├── trotter{order}_N{N}_depth{depth}[_chi{chi}].csv
        └── tepai_d{d}_S{n_circuits}_tstart{tstart}[_chi{chi}].csv

For ``snapshot`` experiments both files are written independently, so
altering only Trotter or only TE-PAI parameters triggers only the
affected component to recompute on the next run.

The folder captures physics parameters shared by all simulations in it
(system size, coupling, observable, etc.).  The filename encodes the
method-specific parameters (Trotter steps, TE-PAI delta exponent, …).

Each ``.csv`` file begins with ``# key=value`` metadata header lines
(one per parameter), followed by a single CSV header row and one row
per snapshot.  The metadata header is used on load to detect stale
caches — if the cached metadata disagrees with the requested
parameters, the run is repeated from scratch.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

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

    Format: ``trotter{order}_N{N}_depth{depth}[_chi{chi}].csv``

    *chi* is included only when ``max_bond`` is set (tensor-network run).
    """
    order = params.get("trotter_order", 1)
    N = params["N"]
    depth = params.get("depth", 1)
    chi = params.get("max_bond")

    name = f"trotter{order}_N{N}_depth{depth}"
    if chi is not None:
        name += f"_chi{chi}"
    return name + ".csv"


def tepai_filename(params: dict[str, Any]) -> str:
    r"""Build the TE-PAI data filename.

    Format: ``tepai_d{d}_S{n_circuits}_tstart{tstart}[_chi{chi}].csv``

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
    return name + ".csv"


def circuits_filename(params: dict[str, Any]) -> str:
    """Build the sampled-circuits filename.

    Format: ``circuits_d{d}_S{n_circuits}_T{T}.csv``
    """
    pod = params["pi_over_delta"]
    d = round(math.log2(pod))
    if 2**d != pod:
        raise ValueError(
            f"pi_over_delta must be a power of 2, got {pod} "
            f"(nearest: 2^{d} = {2**d})"
        )
    n_circuits = params.get("n_circuits", 1)
    T = params["total_time"]
    return f"circuits_d{d}_S{n_circuits}_T{_fmt(T)}.csv"


# ── full path resolution ────────────────────────────────────────────────

def resolve_data_path(params: dict[str, Any]) -> Path:
    """Return the full relative path for the experiment's cached data file.

    Combines :func:`experiment_folder_name` with the method-specific
    filename (Trotter, TE-PAI, or raw circuits).

    For ``snapshot`` experiments the Trotter component path is returned
    for display/logging purposes; use :func:`is_cached` to check whether
    both components exist.
    """
    folder = experiment_folder_name(params)
    exp_type = params.get("type", "")
    if exp_type in ("trotter", "snapshot"):
        fname = trotter_filename(params)
    elif exp_type == "tepai":
        fname = tepai_filename(params)
    elif exp_type == "circuits":
        fname = circuits_filename(params)
    else:
        raise ValueError(f"Unknown experiment type {exp_type!r}")
    return DATA_ROOT / folder / fname


def resolve_data_folder(params: dict[str, Any]) -> Path:
    """Return the data folder for *params* (no filename)."""
    return DATA_ROOT / experiment_folder_name(params)


def is_cached(params: dict[str, Any]) -> bool:
    """Return True if cached data exists on disk for *params*.

    For ``snapshot`` experiments both the Trotter and TE-PAI component
    files must exist for the result to be considered fully cached.
    """
    if params.get("type") == "snapshot":
        trotter_path = resolve_data_path({**params, "type": "trotter"})
        tepai_path = resolve_data_path({**params, "type": "tepai"})
        return trotter_path.exists() and tepai_path.exists()
    return resolve_data_path(params).exists()


# ── CSV I/O ─────────────────────────────────────────────────────────────

def save_csv(
    path: str | Path,
    metadata: dict[str, Any],
    columns: Sequence[str],
    arrays: Sequence[Iterable[float]],
) -> Path:
    """Write a CSV with ``# key=value`` header lines.

    Parameters
    ----------
    path :
        Destination file path.  Parent directory is created as needed.
    metadata :
        Dict of parameters to embed as header comments.  Values are
        stringified via ``str()``.
    columns :
        Column names for the CSV body.
    arrays :
        One iterable per column; all must have the same length.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for k, v in metadata.items():
            f.write(f"# {k}={v}\n")
        f.write(",".join(columns) + "\n")
        for row in zip(*arrays, strict=True):
            f.write(",".join(f"{v}" for v in row) + "\n")
    return path


def load_csv(path: str | Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Read a CSV written by :func:`save_csv`.

    Returns ``(metadata, {col_name: np.array})``.  Metadata values are
    always returned as strings — use :func:`check_metadata` for typed
    comparison.
    """
    meta: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    k, v = body.split("=", 1)
                    meta[k.strip()] = v.strip()
            elif header is None:
                header = [c.strip() for c in line.split(",")]
            else:
                rows.append([float(x) for x in line.split(",")])
    if header is None:
        raise ValueError(f"No CSV header found in {path}")
    arr = np.array(rows) if rows else np.empty((0, len(header)))
    data = {col: arr[:, i] for i, col in enumerate(header)}
    return meta, data


def check_metadata(
    cached: dict[str, str],
    expected: dict[str, Any],
) -> list[tuple[str, str, Any]]:
    """Return a list of ``(key, cached_value, expected_value)`` mismatches.

    Values are compared as floats first (rounded to 10 decimal places to
    absorb repr noise) and fall back to string equality otherwise.  Keys
    missing from *cached* are reported with cached value ``"<missing>"``.
    """
    mismatches: list[tuple[str, str, Any]] = []
    for key, want in expected.items():
        got = cached.get(key)
        if got is None:
            mismatches.append((key, "<missing>", want))
            continue
        try:
            g = float(got)
            w = float(want)
            if round(g, 10) != round(w, 10):
                mismatches.append((key, got, want))
        except (ValueError, TypeError):
            if str(got) != str(want):
                mismatches.append((key, got, want))
    return mismatches


# ── experiment-level convenience ────────────────────────────────────────

def save_results(
    params: dict[str, Any],
    *,
    columns: Sequence[str],
    arrays: Sequence[Iterable[float]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save experiment results as a CSV in the canonical data path.

    Parameters
    ----------
    params :
        Merged parameter dict — used to compute the data path.
    columns :
        Column names for the CSV body.
    arrays :
        One iterable per column.
    metadata :
        Header metadata.  When *None*, ``params`` itself is used.
    """
    path = resolve_data_path(params)
    if metadata is None:
        metadata = params
    return save_csv(path, metadata, columns, arrays)


def load_results(
    params: dict[str, Any],
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Load cached CSV results for *params*.

    Returns ``(metadata, {col: np.array})``.

    Raises
    ------
    FileNotFoundError
        If no cached file exists.
    """
    path = resolve_data_path(params)
    if not path.exists():
        raise FileNotFoundError(f"No cached data at {path}")
    return load_csv(path)
