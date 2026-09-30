import numpy as np

from ofdm_link.ofdm import OFDMConfig, demodulate_ofdm, modulate_ofdm


def test_ofdm_round_trip() -> None:
    rng = np.random.default_rng(2)
    config = OFDMConfig(n_fft=64, cp_len=16)
    symbols = rng.normal(size=64) + 1j * rng.normal(size=64)
    recovered = demodulate_ofdm(modulate_ofdm(symbols, config), config)
    assert np.allclose(symbols, recovered)
