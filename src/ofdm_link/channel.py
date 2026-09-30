from __future__ import annotations

import numpy as np


def _complex_noise(shape, noise_power: float, rng: np.random.Generator) -> np.ndarray:
    sigma = np.sqrt(noise_power / 2.0)
    return sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return signal + _complex_noise(signal.shape, noise_power, rng)


def flat_rayleigh(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, complex]:
    """Apply one flat Rayleigh coefficient and AWGN referenced to transmit power."""
    signal = np.asarray(signal, dtype=np.complex128)
    h = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    faded = h * signal
    return faded + _complex_noise(signal.shape, noise_power, rng), complex(h)
