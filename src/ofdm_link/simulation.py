from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .adaptive import choose_profile
from .channel import awgn, flat_rayleigh
from .modulation import demodulate, modulate
from .ofdm import OFDMConfig, demodulate_ofdm, modulate_ofdm


@dataclass(frozen=True, slots=True)
class SweepConfig:
    snr_db: list[float]
    modulation_orders: list[int]
    channel: str = "awgn"
    frames_per_point: int = 100
    n_fft: int = 64
    cp_len: int = 16
    seed: int = 7


def _frame(
    order: int,
    snr_db: float,
    channel: str,
    config: OFDMConfig,
    rng: np.random.Generator,
) -> tuple[int, int]:
    bits_per_symbol = order.bit_length() - 1
    bits = rng.integers(0, 2, size=config.n_fft * bits_per_symbol, dtype=np.uint8)
    freq_symbols = modulate(bits, order)
    tx = modulate_ofdm(freq_symbols, config)

    if channel == "awgn":
        rx = awgn(tx, snr_db, rng)
    elif channel == "rayleigh":
        rx_faded, h = flat_rayleigh(tx, snr_db, rng)
        if abs(h) < 1e-12:
            return len(bits), len(bits)
        rx = rx_faded / h
    else:
        raise ValueError("channel must be 'awgn' or 'rayleigh'")

    detected = demodulate(demodulate_ofdm(rx, config), order)
    errors = int(np.count_nonzero(bits != detected))
    return errors, len(bits)


def run_snr_sweep(config: SweepConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    ofdm = OFDMConfig(config.n_fft, config.cp_len)
    curves: dict[str, list[dict[str, float]]] = {}

    for order in config.modulation_orders:
        points: list[dict[str, float]] = []
        for snr in config.snr_db:
            errors = total = 0
            for _ in range(config.frames_per_point):
                frame_errors, frame_total = _frame(order, snr, config.channel, ofdm, rng)
                errors += frame_errors
                total += frame_total
            points.append({"snr_db": float(snr), "ber": errors / total if total else 0.0})
        curves[str(order)] = points

    adaptive = [
        {
            "snr_db": float(snr),
            "modulation": choose_profile(snr).name,
            "order": choose_profile(snr).order,
            "bits_per_symbol": choose_profile(snr).bits_per_symbol,
        }
        for snr in config.snr_db
    ]
    return {
        "config": {
            "channel": config.channel,
            "frames_per_point": config.frames_per_point,
            "n_fft": config.n_fft,
            "cp_len": config.cp_len,
            "seed": config.seed,
        },
        "curves": curves,
        "adaptive_policy": adaptive,
    }
