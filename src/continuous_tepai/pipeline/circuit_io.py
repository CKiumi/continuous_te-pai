"""Compact CSV serialization of sampled TE-PAI circuits.

Each circuit becomes one line in a CSV with the layout::

    <circuit_id>,<length>,<pi_parity>,<gate1>,<gate2>,...

Gate encoding rules:

* Pauli letters for every non-identity position on the Pauli string,
  followed by the qubit indices joined by ``a`` (for "and").
* Uppercase letters for ``+Δ`` rotations, lowercase for ``−Δ``.
* A π rotation uses lowercase letters with ``!`` appended immediately
  after the letters (before the qubit indices).

Examples:
    ``XX0a1``  — ``+Δ`` rotation of ``X⊗X`` on qubits 0 and 1.
    ``xx0a1``  — ``−Δ`` rotation of the same term.
    ``xx!0a1`` — π rotation of the same term.
    ``Z3``     — ``+Δ`` rotation of ``Z`` on qubit 3.

The metadata header consists of ``# key=value`` comment lines; no CSV
column header row is written because circuit rows have variable length.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ..circuit import SampledCircuit


@lru_cache(maxsize=4096)
def _encode_pauli(label: str) -> tuple[str, str]:
    """Split a Pauli-string label into (letters, qubit_str).

    Cached because a single Hamiltonian reuses the same ``L`` Pauli
    strings across millions of sampled rotations.
    """
    letters: list[str] = []
    qubits: list[str] = []
    for i, ch in enumerate(label):
        if ch != "I":
            letters.append(ch)
            qubits.append(str(i))
    return "".join(letters), "a".join(qubits)


_PI = math.pi


def _encode_gate(label: str, angle: float) -> str:
    letters, qubits = _encode_pauli(label)
    if abs(abs(angle) - _PI) < 1e-9:
        return letters.lower() + "!" + qubits
    if angle >= 0.0:
        return letters.upper() + qubits
    return letters.lower() + qubits


def serialize_circuit(cid: int, circuit: SampledCircuit) -> str:
    """Return the one-line CSV encoding of *circuit*."""
    rots = circuit.rotations
    pi_parity = 0
    gate_tokens: list[str] = []
    for r in rots:
        angle = r.angle
        if abs(abs(angle) - _PI) < 1e-9:
            pi_parity ^= 1
        gate_tokens.append(_encode_gate(r.pauli.label, angle))
    head = f"{cid},{len(rots)},{pi_parity}"
    if not gate_tokens:
        return head
    return head + "," + ",".join(gate_tokens)


def save_circuits_csv(
    path: str | Path,
    circuits: Sequence[SampledCircuit],
    metadata: dict[str, Any],
) -> Path:
    """Write *circuits* to *path* with commented metadata header.

    Parameters
    ----------
    path :
        Destination CSV path.  Parent directory is created as needed.
    circuits :
        Sequence of :class:`SampledCircuit` in the order they were drawn.
    metadata :
        ``# key=value`` header block — should include Δ, T, n_qubits,
        n_circuits, seed, and any Hamiltonian parameters needed to
        reconstruct the sampling configuration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf: list[str] = [f"# {k}={v}\n" for k, v in metadata.items()]
    buf.extend(serialize_circuit(i, c) + "\n" for i, c in enumerate(circuits))
    with path.open("w") as f:
        f.writelines(buf)
    return path
