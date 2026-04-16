# Continuous TE-PAI

## Features

- Implements **Continuous TE-PAI**, a randomized Hamiltonian simulation protocol that applies probabilistic angle interpolation directly to continuous time evolution — eliminating discretization error with finite classical and quantum resources.
- Supports **time-dependent Hamiltonians** of the form H(t) = Σ_k c_k(t) P_k.
- **Backend-agnostic design**: swap between Qiskit, Qulacs, or an MPS (quimb) backend without changing algorithm code.

---

## 📦 Installation

### Install with `uv` (recommended)

Sync core dependencies:

```bash
uv sync
```

Include a backend of your choice (mix-and-match):

```bash
uv sync --extra qiskit   # Qiskit statevector / primitives
uv sync --extra qulacs   # Qulacs C++ statevector
uv sync --extra mps      # quimb-based Matrix Product State backend
```

### Install with `pip`

```bash
pip install -e .
pip install -e ".[mps]"  # MPS extra
```

---

## 🧮 MPS backend

The MPS backend uses [`quimb`](https://quimb.readthedocs.io/) to run TE-PAI
circuits as a matrix-product state, so it can reach far more qubits than
dense statevector simulation whenever the state stays close to
low-entanglement.  It plugs into the same `Backend` protocol as the
Qiskit / Qulacs backends, so you just swap the class:

```python
import numpy as np
from continuous_tepai import ContinuousTEPAI, Hamiltonian, PauliString
from continuous_tepai.backends import MPSBackend

H = Hamiltonian.from_local_terms(4, [
    ("ZZ", [0, 1], lambda t: 1.0),
    ("ZZ", [1, 2], lambda t: 1.0),
    ("ZZ", [2, 3], lambda t: 1.0),
    ("X",  [0],    lambda t: np.cos(t)),
])

sampler = ContinuousTEPAI(H, delta=0.3, total_time=0.5, seed=0)
circuits = sampler.sample_circuits(64)

mps = MPSBackend()              # exact (no truncation) — use max_bond=... to cap
# mps = MPSBackend(max_bond=64) # truncate for larger systems

obs = PauliString("ZIII")
estimate = np.mean([
    c.weight * mps.expectation(c.rotations, obs, H.num_qubits, initial_state="plus")
    for c in circuits
])
```

A runnable example lives in `examples/mps_demo.py`:

```bash
uv run python examples/mps_demo.py
```

### Supported rotations / observables

The backend accepts the Pauli rotations actually produced by the Continuous
TE-PAI sampler:

* single-qubit rotations `R_X, R_Y, R_Z`;
* two-qubit same-type rotations `R_XX, R_YY, R_ZZ`.

Mixed two-qubit rotations (e.g. `XZ`) or ≥3-body rotations raise a
`NotImplementedError`.  Observables may be any Pauli string — single-site
calls take the short MPS path; multi-site Pauli strings are evaluated via
`⟨ψ|P|ψ⟩` applied directly on the MPS.

---

## 🧪 Testing with `pytest`

All test files are located in the `tests/` directory.

1. **Sync dependencies (including extras):**

```bash
   uv sync --all-extras
```

2. **Run tests:**

```bash
   uv run pytest -s
```

MPS tests (including parity against Qulacs) live in
`tests/test_mps_backend.py` and are skipped automatically if `quimb` is
not installed.