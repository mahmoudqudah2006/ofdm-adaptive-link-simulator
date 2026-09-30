from __future__ import annotations

import argparse
import json
from pathlib import Path

from .simulation import SweepConfig, run_snr_sweep


def _snr_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    value = start
    while value <= stop + 1e-12:
        values.append(round(value, 10))
        value += step
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="OFDM adaptive link simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="Run a BER-vs-SNR sweep")
    sweep.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    sweep.add_argument("--snr-start", type=float, default=0.0)
    sweep.add_argument("--snr-stop", type=float, default=24.0)
    sweep.add_argument("--snr-step", type=float, default=2.0)
    sweep.add_argument("--frames", type=int, default=100)
    sweep.add_argument("--orders", type=int, nargs="+", default=[2, 4, 16, 64])
    sweep.add_argument("--seed", type=int, default=7)
    sweep.add_argument("--output", type=Path, default=Path("results/ofdm_sweep.json"))
    args = parser.parse_args()

    result = run_snr_sweep(
        SweepConfig(
            snr_db=_snr_values(args.snr_start, args.snr_stop, args.snr_step),
            modulation_orders=args.orders,
            channel=args.channel,
            frames_per_point=args.frames,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
