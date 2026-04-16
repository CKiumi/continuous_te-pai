import pytest
from continuous_tepai import PauliString


def test_valid():
    p = PauliString("XIZI")
    assert p.num_qubits == 4


def test_invalid():
    with pytest.raises(ValueError):
        PauliString("ABC")


def test_frozen():
    p = PauliString("XY")
    with pytest.raises(AttributeError):
        p.label = "ZZ"