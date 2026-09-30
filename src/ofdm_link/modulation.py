from __future__ import annotations

import math

import numpy as np


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    array = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((array != 0) & (array != 1)):
        raise ValueError("bits must contain only 0 and 1")
    return array


def _gray_to_binary(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.int64, copy=True)
    result = value.copy()
    shift = 1
    while np.any(value >> shift):
        result ^= value >> shift
        shift += 1
    return result


def _binary_to_gray(value: np.ndarray) -> np.ndarray:
    return value ^ (value >> 1)


def _groups_to_int(groups: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(groups.shape[1] - 1, -1, -1)
    return groups @ weights


def _int_to_groups(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


def modulate(bits: np.ndarray, order: int) -> np.ndarray:
    """Map bits to unit-average-power BPSK or square Gray-coded QAM symbols."""
    bits = _validate_bits(bits)
    if order == 2:
        if len(bits) == 0:
            return np.array([], dtype=np.complex128)
        return (2.0 * bits.astype(float) - 1.0).astype(np.complex128)

    root = int(math.isqrt(order))
    if root * root != order or root not in {2, 4, 8}:
        raise ValueError("order must be one of 2, 4, 16, 64")

    bits_per_symbol = int(math.log2(order))
    if len(bits) % bits_per_symbol:
        raise ValueError("bit length must be divisible by log2(order)")

    axis_bits = bits_per_symbol // 2
    groups = bits.reshape(-1, bits_per_symbol)
    i_gray = _groups_to_int(groups[:, :axis_bits])
    q_gray = _groups_to_int(groups[:, axis_bits:])
    i_binary = _gray_to_binary(i_gray)
    q_binary = _gray_to_binary(q_gray)
    i_level = 2 * i_binary - (root - 1)
    q_level = 2 * q_binary - (root - 1)
    norm = math.sqrt((2.0 / 3.0) * (order - 1))
    return (i_level + 1j * q_level) / norm


def demodulate(symbols: np.ndarray, order: int) -> np.ndarray:
    """Hard-decision demodulation matching :func:`modulate`."""
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if order == 2:
        return (symbols.real >= 0).astype(np.uint8)

    root = int(math.isqrt(order))
    if root * root != order or root not in {2, 4, 8}:
        raise ValueError("order must be one of 2, 4, 16, 64")

    bits_per_symbol = int(math.log2(order))
    axis_bits = bits_per_symbol // 2
    norm = math.sqrt((2.0 / 3.0) * (order - 1))
    levels = np.arange(-(root - 1), root, 2, dtype=float)

    scaled_i = symbols.real * norm
    scaled_q = symbols.imag * norm
    i_binary = np.argmin(np.abs(scaled_i[:, None] - levels[None, :]), axis=1)
    q_binary = np.argmin(np.abs(scaled_q[:, None] - levels[None, :]), axis=1)
    i_gray = _binary_to_gray(i_binary)
    q_gray = _binary_to_gray(q_binary)
    i_bits = _int_to_groups(i_gray, axis_bits)
    q_bits = _int_to_groups(q_gray, axis_bits)
    return np.concatenate([i_bits, q_bits], axis=1).reshape(-1)
