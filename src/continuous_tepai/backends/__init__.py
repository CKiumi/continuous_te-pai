"""Built-in backends for Continuous TE-PAI."""


def __getattr__(name: str):
    if name == "QiskitBackend":
        from .qiskit_backend import QiskitBackend
        return QiskitBackend
    if name == "QulacsBackend":
        from .qulacs_backend import QulacsBackend
        return QulacsBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["QiskitBackend", "QulacsBackend"]