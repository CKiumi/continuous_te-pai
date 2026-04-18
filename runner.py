#!/usr/bin/env python3
"""Run active experiments defined in config.json.

Usage::

    uv run python runner.py                  # run active experiments
    uv run python runner.py --config my.json # alternate config file
    uv run python runner.py --dry-run        # show what would run, skip computation

The pipeline for each active experiment is:

1. Merge ``defaults`` with per-experiment overrides.
2. Compute the canonical data path from parameters.
3. If the ``.npz`` file already exists → load cached data and skip.
4. Otherwise → run the experiment → save results → write ``data_path``
   back into the config so subsequent runs find the cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from continuous_tepai.pipeline.config import load_config, merge_params, set_data_path
from continuous_tepai.pipeline.cache import resolve_data_path, is_cached, save_results, load_results
from continuous_tepai.pipeline.experiment import run_experiment

log = logging.getLogger("runner")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run active TE-PAI / Trotter experiments.")
    parser.add_argument(
        "--config", default="config.json",
        help="Path to the JSON config file (default: config.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing experiments.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-8s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    defaults = config.get("defaults", {})
    experiments = config.get("experiments", [])

    active = [e for e in experiments if e.get("active", False)]
    if not active:
        log.info("No active experiments in %s — nothing to do.", args.config)
        return

    log.info(
        "Found %d active experiment(s) out of %d total.",
        len(active), len(experiments),
    )

    for exp in active:
        name = exp.get("name", "<unnamed>")
        exp_type = exp.get("type", "?")
        params = merge_params(defaults, exp)

        log.info("─" * 60)
        log.info("Experiment: %s  [type=%s]", name, exp_type)

        try:
            data_path = resolve_data_path(params)
        except (ValueError, KeyError) as exc:
            log.error("  Skipping %s: cannot resolve data path — %s", name, exc)
            continue

        log.info("  Data path: %s", data_path)

        if is_cached(params):
            log.info("  ✓ Cached data found — skipping computation.")
            if args.dry_run:
                continue
            if exp_type != "snapshot":
                metadata, arrays = load_results(params)
                _log_summary(arrays)
            continue

        if args.dry_run:
            log.info("  [dry-run] Would run computation → save to %s", data_path)
            continue

        # Run the experiment
        log.info("  Running computation …")
        t0 = time.perf_counter()
        try:
            arrays = run_experiment(params)
        except NotImplementedError as exc:
            log.warning("  ⚠ %s — skipping.", exc)
            continue
        except Exception:
            log.exception("  ✗ Experiment %s failed:", name)
            continue
        elapsed = time.perf_counter() - t0
        log.info("  Done in %.1f s.", elapsed)

        # Save results (snapshot saves its components internally in run_snapshot)
        if exp_type != "snapshot":
            saved_path = save_results(
                params,
                columns=list(arrays.keys()),
                arrays=list(arrays.values()),
            )
            log.info("  Saved to %s", saved_path)
            set_data_path(config, name, str(saved_path), args.config)
            log.info("  Updated config with data_path.")

        _log_summary(arrays)

    log.info("─" * 60)
    log.info("All active experiments processed.")


def _log_summary(arrays: dict) -> None:
    """Print a one-line summary of the cached / computed arrays."""
    parts = []
    for k, v in arrays.items():
        if hasattr(v, "shape"):
            parts.append(f"{k}{list(v.shape)}")
        else:
            parts.append(k)
    if parts:
        log.info("  Arrays: %s", ", ".join(parts))


if __name__ == "__main__":
    main()
