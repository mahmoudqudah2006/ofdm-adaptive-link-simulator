from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModulationProfile:
    name: str
    order: int
    minimum_snr_db: float

    @property
    def bits_per_symbol(self) -> int:
        return self.order.bit_length() - 1


DEFAULT_PROFILES = (
    ModulationProfile("BPSK", 2, float("-inf")),
    ModulationProfile("QPSK", 4, 4.0),
    ModulationProfile("16-QAM", 16, 10.0),
    ModulationProfile("64-QAM", 64, 18.0),
)


def choose_profile(snr_db: float) -> ModulationProfile:
    selected = DEFAULT_PROFILES[0]
    for profile in DEFAULT_PROFILES:
        if snr_db >= profile.minimum_snr_db:
            selected = profile
    return selected
