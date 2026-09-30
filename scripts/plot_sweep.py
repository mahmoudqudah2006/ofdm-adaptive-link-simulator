from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ber_curve.png"))
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 5))
    for order, points in payload["curves"].items():
        ax.semilogy(
            [p["snr_db"] for p in points],
            [max(p["ber"], 1e-6) for p in points],
            marker="o",
            label=f"M={order}",
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title(f"OFDM BER — {payload['config']['channel'].upper()} channel")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
