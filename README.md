# Continuous TE-PAI

## Features

- Implements **Continuous TE-PAI**, a randomized Hamiltonian simulation protocol that applies probabilistic angle interpolation directly to continuous time evolution — eliminating discretization error with finite classical and quantum resources.
- Supports **time-dependent Hamiltonians** of the form H(t) = Σ_k c_k(t) P_k.
- **Backend-agnostic design**: swap between Qiskit or custom backends without changing algorithm code.

---

## 📦 Installation

### Install with `uv` (recommended)

Sync core dependencies:

```bash
uv sync
```

Include the Qiskit backend:

```bash
uv sync --extra qiskit
```

### Install with `pip`

```bash
pip install -e .
```

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