from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OFDMConfig:
    n_fft: int = 64
    cp_len: int = 16

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if not 0 <= self.cp_len < self.n_fft:
            raise ValueError("cp_len must satisfy 0 <= cp_len < n_fft")


def modulate_ofdm(symbols: np.ndarray, config: OFDMConfig) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if len(symbols) != config.n_fft:
        raise ValueError("one OFDM symbol requires exactly n_fft frequency-domain symbols")
    time_domain = np.fft.ifft(symbols) * np.sqrt(config.n_fft)
    if config.cp_len == 0:
        return time_domain
    return np.concatenate([time_domain[-config.cp_len :], time_domain])


def demodulate_ofdm(samples: np.ndarray, config: OFDMConfig) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
    expected = config.n_fft + config.cp_len
    if len(samples) != expected:
        raise ValueError(f"expected {expected} samples")
    payload = samples[config.cp_len :]
    return np.fft.fft(payload) / np.sqrt(config.n_fft)
