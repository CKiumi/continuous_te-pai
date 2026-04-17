"""Configuration loading, merging, and persistence.

The config file (``config.json`` by default) has this shape::

    {
      "defaults": { ... },        # shared default parameters
      "experiments": [             # list of experiment definitions
        {
          "name": "experiment_1",
          "active": true,
          "type": "trotter",       # or "tepai"
          ...                      # any default can be overridden here
        },
        ...
      ]
    }

After a successful run the experiment entry gains a ``data_path`` field
that points to the cached ``.npz`` file so subsequent runs can skip the
computation.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    """Read the JSON config and return the parsed dict.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def save_config(config: dict[str, Any], path: str | Path = "config.json") -> None:
    """Write *config* back to *path* with pretty-print formatting.

    The file is overwritten atomically (write → rename) so a crash
    mid-write does not corrupt the config.
    """
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def merge_params(defaults: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    """Return a flat parameter dict: *defaults* overridden by *experiment*.

    Keys that are experiment-management metadata (``name``, ``active``,
    ``note``, ``type``, ``data_path``) are included verbatim from the
    experiment entry but do **not** override physics defaults.
    """
    merged = copy.deepcopy(defaults)
    for key, value in experiment.items():
        merged[key] = value
    return merged


def set_data_path(
    config: dict[str, Any],
    experiment_name: str,
    data_path: str,
    config_path: str | Path = "config.json",
) -> None:
    """Write *data_path* into the experiment entry and persist the config.

    Parameters
    ----------
    config :
        The full config dict (will be mutated in place).
    experiment_name :
        Value of the ``"name"`` field that identifies the experiment.
    data_path :
        Relative path (from project root) to the cached ``.npz`` file.
    config_path :
        Where to write the updated config.
    """
    for exp in config["experiments"]:
        if exp["name"] == experiment_name:
            exp["data_path"] = data_path
            break
    else:
        raise KeyError(f"No experiment named {experiment_name!r} in config")
    save_config(config, config_path)
