import numpy as np
import pytest

from ofdm_link.modulation import demodulate, modulate


@pytest.mark.parametrize("order", [2, 4, 16, 64])
def test_modulation_round_trip(order: int) -> None:
    rng = np.random.default_rng(3)
    bps = order.bit_length() - 1
    bits = rng.integers(0, 2, size=100 * bps, dtype=np.uint8)
    recovered = demodulate(modulate(bits, order), order)
    assert np.array_equal(bits, recovered)


def test_average_power_is_near_one() -> None:
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=60000, dtype=np.uint8)
    symbols = modulate(bits, 64)
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0, rel=0.03)
