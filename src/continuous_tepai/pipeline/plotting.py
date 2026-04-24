"""Plot generation for the numerics pipeline.

Produces publication-quality comparison plots of Trotter reference vs
Continuous TE-PAI estimates, matching the style of ``examples/snapshot.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def snapshot_comparison_plot(
    times: np.ndarray,
    trotter_values: np.ndarray,
    tepai_mean: np.ndarray,
    tepai_std: np.ndarray,
    *,
    n_qubits: int,
    observable_str: str = "X0",
    n_circuits: int | None = None,
    output_path: str | Path = "snapshot.pdf",
) -> Path:
    r"""Generate a Trotter-vs-TE-PAI comparison plot and save to *output_path*.

    The plot shows:
    * A solid line for the Trotter reference.
    * Error-bar markers for the Continuous TE-PAI estimate.

    Parameters
    ----------
    times :
        Snapshot times (shared x-axis), shape ``(n_snapshots + 1,)``.
    trotter_values :
        Trotter expectation values, same shape as *times*.
    tepai_mean :
        TE-PAI mean estimates, same shape as *times*.
    tepai_std :
        TE-PAI standard errors, same shape as *times*.
    n_qubits :
        System size (used in the title).
    observable_str :
        Short label for the observable (e.g. ``"X0"``), used in the title.
    n_circuits :
        Number of TE-PAI circuits (shown in the legend).
    output_path :
        Where to save the PDF.

    Returns
    -------
    Path
        The absolute path of the saved file.
    """
    # Import matplotlib only when needed so the rest of the pipeline
    # works without a display or matplotlib installed.
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "lines.linewidth": 2.5,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Trotter reference
    ax.plot(times, trotter_values, "k-", alpha=0.7, label="Trotter reference")

    # TE-PAI estimate with error bars
    tepai_label = "Continuous TE-PAI"
    if n_circuits is not None:
        tepai_label += rf" ($N_s$={n_circuits})"
    _, caps, bars = ax.errorbar(
        times,
        tepai_mean,
        yerr=tepai_std,
        fmt="b--x",
        label=tepai_label,
        capsize=5,
    )
    for bar in bars:
        bar.set_alpha(0.5)
    for cap in caps:
        cap.set_alpha(0.5)

    # Formatting
    pauli_char = observable_str[0]
    qubit_idx = observable_str[1:]
    ax.set_title(
        rf"$\langle {pauli_char}_{{{qubit_idx}}} \rangle$"
        f"  —  {n_qubits}-qubit spin chain"
    )
    ax.set_xlabel(r"Time $T$")
    ax.set_ylabel("Expectation value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def bond_tracking_plot(
    times: np.ndarray,
    trotter_values: np.ndarray,
    tepai_mean: np.ndarray,
    tepai_std: np.ndarray,
    trotter_bonds: np.ndarray,
    tepai_max_bonds: np.ndarray,
    *,
    n_qubits: int,
    observable_str: str = "X0",
    n_circuits: int | None = None,
    max_bond_cap: int | None = None,
    trotter_gate_count: int | None = None,
    tepai_avg_gate_count: float | None = None,
    trotter_final_bond: int | None = None,
    tepai_final_bonds: np.ndarray | None = None,
    output_path: str | Path = "bond_tracking.pdf",
) -> Path:
    r"""Two-panel plot: expectation values (top) and MPS max-bond (bottom).

    Top panel matches :func:`snapshot_comparison_plot` — Trotter reference
    as a solid black line and Continuous TE-PAI as blue dashed errorbars.
    Bottom panel shows the per-snapshot MPS max-bond for Trotter (black,
    solid) and the worst-case max-bond across all TE-PAI circuits at each
    snapshot (blue, dashed).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "lines.linewidth": 2.5,
        }
    )

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 2])
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_hist = fig.add_subplot(gs[2])

    # ── top: expectation values ───────────────────────────────────
    ax_top.plot(times, trotter_values, "k-", alpha=0.7, label="Trotter reference")
    tepai_label = "Continuous TE-PAI"
    if n_circuits is not None:
        tepai_label += rf" ($N_s$={n_circuits})"
    _, caps, bars = ax_top.errorbar(
        times, tepai_mean, yerr=tepai_std,
        fmt="b--x", label=tepai_label, capsize=5,
    )
    for bar in bars:
        bar.set_alpha(0.5)
    for cap in caps:
        cap.set_alpha(0.5)

    pauli_char = observable_str[0]
    qubit_idx = observable_str[1:]
    title = (
        rf"$\langle {pauli_char}_{{{qubit_idx}}} \rangle$"
        f"  —  {n_qubits}-qubit spin chain"
    )
    if max_bond_cap is not None:
        title += rf"  ($\chi_{{\max}}={max_bond_cap}$)"
    ax_top.set_title(title)
    ax_top.set_ylabel("Expectation value")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    # ── bottom: max-bond over time ────────────────────────────────
    ax_bot.plot(times, trotter_bonds, "k-")
    ax_bot.plot(times, tepai_max_bonds, "b--")
    if max_bond_cap is not None:
        ax_bot.axhline(
            max_bond_cap, color="gray", linestyle=":", linewidth=1.5,
        )
    ax_bot.set_xlabel(r"Time $T$")
    ax_bot.set_ylabel(r"MPS max bond $\chi$")
    ax_bot.grid(alpha=0.3)

    # Gate-count annotations along the top of the bottom panel.
    if trotter_gate_count is not None:
        ax_bot.text(
            0.02, 0.97,
            f"Trotter gates: {trotter_gate_count}",
            transform=ax_bot.transAxes,
            ha="left", va="top",
            color="black", fontsize=13,
        )
    if tepai_avg_gate_count is not None and not np.isnan(tepai_avg_gate_count):
        ax_bot.text(
            0.98, 0.97,
            f"TE-PAI avg gates: {tepai_avg_gate_count:.1f}",
            transform=ax_bot.transAxes,
            ha="right", va="top",
            color="blue", fontsize=13,
        )

    # ── histogram: per-circuit max-bond at final snapshot ─────────
    if tepai_final_bonds is not None and len(tepai_final_bonds) > 0:
        lo = int(min(tepai_final_bonds.min(),
                     trotter_final_bond if trotter_final_bond is not None else 10**9))
        hi = int(max(tepai_final_bonds.max(),
                     trotter_final_bond if trotter_final_bond is not None else 0))
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        ax_hist.hist(
            tepai_final_bonds, bins=bins, color="blue",
            alpha=0.6, edgecolor="blue",
        )
    if trotter_final_bond is not None:
        ax_hist.axvline(trotter_final_bond, color="black", linewidth=2.5)
    ax_hist.set_xlabel(r"$\chi$")
    ax_hist.set_ylabel("count")
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def overhead_plot(
    times: np.ndarray,
    overheads_by_n: dict[int, np.ndarray],
    *,
    pi_over_delta: int,
    j: float,
    time_dependent: bool,
    output_path: str | Path = "overhead.pdf",
) -> Path:
    r"""Plot TE-PAI overhead vs time for several qubit counts.

    The overhead is :math:`\exp(2\,\|c\|_{1,\text{avg}}\,T\,\tan(\Delta/2))`
    — the exponential prefactor on every sampled-circuit weight.

    Parameters
    ----------
    times :
        Time grid shared by all curves, shape ``(n_snapshots + 1,)``.
    overheads_by_n :
        Mapping ``n_qubits → overhead(t)`` array.
    pi_over_delta :
        The value :math:`\pi / \Delta` used (shown in the title).
    j :
        Coupling strength (shown in the title).
    time_dependent :
        Whether :math:`J(t)` is time-dependent (shown in the title).
    output_path :
        Where to save the PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "lines.linewidth": 2.5,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    qubit_counts = sorted(overheads_by_n.keys())
    cmap = plt.get_cmap("viridis")
    for idx, n in enumerate(qubit_counts):
        color = cmap(idx / max(len(qubit_counts) - 1, 1))
        ax.plot(times, overheads_by_n[n], label=f"n = {n}", color=color)

    j_str = (
        rf"$J(t)=\cos({j:g}\pi t)$" if time_dependent else rf"$J={j:g}$"
    )
    ax.set_title(
        rf"TE-PAI overhead  —  $\Delta=\pi/{pi_over_delta}$, {j_str}"
    )
    ax.set_xlabel(r"Time $T$")
    ax.set_ylabel(r"Overhead  $\exp(2\,\|c\|_{1,\mathrm{avg}}\,T\,\tan(\Delta/2))$")
    ax.legend(title="Qubits")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
