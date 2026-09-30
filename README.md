# OFDM Adaptive Link Simulator

A compact, reproducible **OFDM physical-layer simulator** for studying modulation, channel impairment, BER, and SNR-driven adaptive modulation.

The project is intentionally written as readable research software rather than a black-box communications toolbox. It is suitable for portfolio work, classroom experiments, algorithm prototyping, and as a foundation for later ML/RL-based link adaptation.

## Features

- Baseband OFDM modulation/demodulation with configurable FFT and cyclic prefix
- BPSK, QPSK, 16-QAM, and 64-QAM
- Gray-coded QAM mapping and hard-decision demodulation
- AWGN channel
- Flat Rayleigh fading with perfect one-tap equalization
- BER versus SNR sweeps
- Threshold-based adaptive modulation
- Spectral-efficiency tracking
- Reproducible random seeds
- JSON result export
- Publication-style plotting helper
- Unit tests and GitHub Actions CI

## Signal model

For an OFDM symbol with frequency-domain symbols (X[k]), the transmitted time-domain signal is

[
x[n] = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} X[k]e^{j2\pi kn/N}.
]

A cyclic prefix of length (N_{CP}) is prepended. The AWGN model is

[
y[n] = x[n] + w[n],
]

where the complex Gaussian noise variance is derived from the measured signal power and requested SNR.

For flat Rayleigh fading,

[
y[n] = h x[n] + w[n], \qquad h\sim\mathcal{CN}(0,1),
]

and the simulator applies perfect one-tap equalization (y/h). This is intentionally a link-level abstraction; channel estimation error and frequency-selective multipath are roadmap items.

## Adaptive modulation

The default threshold policy is deliberately simple and inspectable:

| SNR | Modulation | Bits/symbol |
|---:|---:|---:|
| < 4 dB | BPSK | 1 |
| 4–10 dB | QPSK | 2 |
| 10–18 dB | 16-QAM | 4 |
| >= 18 dB | 64-QAM | 6 |

These thresholds are experiment defaults, **not standardized MCS thresholds**. They can be replaced in research experiments or by the companion `rl-link-adaptation` project.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,plot]'

ofdm-link sweep --channel awgn --snr-start 0 --snr-stop 24 --snr-step 2
```

Results are written to `results/ofdm_sweep.json`.

Generate a figure:

```bash
python scripts/plot_sweep.py results/ofdm_sweep.json --output results/ber_curve.png
```

## Python example

```python
from ofdm_link.simulation import SweepConfig, run_snr_sweep

result = run_snr_sweep(
    SweepConfig(
        snr_db=[0, 4, 8, 12, 16, 20, 24],
        modulation_orders=[4, 16, 64],
        channel="awgn",
        frames_per_point=100,
        seed=7,
    )
)
print(result)
```

## Repository structure

```text
src/ofdm_link/
  modulation.py   Gray-coded PSK/QAM mapping
  ofdm.py         FFT/IFFT and cyclic-prefix processing
  channel.py      AWGN and flat Rayleigh models
  adaptive.py     SNR-to-modulation policy
  simulation.py   BER sweeps and experiment orchestration
  cli.py          Command-line interface
scripts/
  plot_sweep.py
tests/
```

## Scope and limitations

This release focuses on a clean baseband simulation. It does not currently model FEC coding, synchronization, CFO, pilot-based channel estimation, PAPR mitigation, or a 3GPP NR resource grid. Those are intentionally left as explicit extensions rather than hidden approximations.

## Roadmap

- [ ] Frequency-selective tapped-delay channels
- [ ] Pilot-based channel estimation
- [ ] Coding and BLER curves
- [ ] CFO and synchronization error
- [ ] PAPR analysis
- [ ] Adaptive coding and modulation
- [ ] RL policy integration
- [ ] 5G NR-inspired numerology experiments

## License

MIT

---

**Mahmoud Alqudah** · Senior Software Engineer · PhD Researcher · Wireless Communications & AI
