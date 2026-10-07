"""Generate a Gaussian-noise PWL voltage source for HSPICE (no TRNOISE needed).

The noise is white, one independent sample per clock (sample interval = 1/fs), rms = sigma.
Output: a text file to `.inc` in the netlist; it defines ONE series source between two nodes.

Usage examples
  python make_noise_pwl.py --snr 80 --n 17000 --seed 1 --out noise_80dB.inc
  python make_noise_pwl.py --sigma 339e-6 --n 17000 --seed 1 --out noise_339u.inc

Options
  --sigma V     rms noise voltage [V]                     (or --snr)
  --snr dB      target in-band noise-only SNR; sigma is computed from
                  sigma^2 = Ps / 10^(snr/10) * (fs/2) / bw
  --ps  V^2     signal power for --snr, default 0.045 (= differential 0.30 V peak)
  --bw  Hz      signal bandwidth for --snr, default 23437.5
  --fs  Hz      sample rate (noise sample interval = 1/fs), default 12e6
  --n   N       number of noise samples (>= N + nskip + a few), default 17000
  --seed S      random seed (same seed -> same file), default 1
  --name/--np/--nn   source name and its two nodes (default vnp vip vip_s)
"""
import argparse
import math

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sigma", type=float)
ap.add_argument("--snr", type=float)
ap.add_argument("--ps", type=float, default=0.045)
ap.add_argument("--bw", type=float, default=23437.5)
ap.add_argument("--fs", type=float, default=12e6)
ap.add_argument("--n", type=int, default=17000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="noise_pwl.inc")
ap.add_argument("--name", default="vnp")
ap.add_argument("--np", dest="np_", default="vip")
ap.add_argument("--nn", default="vip_s")
a = ap.parse_args()

if (a.sigma is None) == (a.snr is None):
    ap.error("give exactly one of --sigma or --snr")
sigma = a.sigma if a.sigma is not None else math.sqrt(a.ps / 10 ** (a.snr / 10) * (a.fs / 2) / a.bw)
T = 1.0 / a.fs
rng = np.random.default_rng(a.seed)
v = rng.normal(0.0, sigma, a.n)
inband = sigma ** 2 * a.bw / (a.fs / 2)               # expected in-band noise power [V^2]

with open(a.out, "w") as f:
    f.write(f"* Gaussian white noise, one sample per clock (T = {T:.6e} s)\n")
    f.write(f"* sigma = {sigma:.6e} V rms, n = {a.n}, seed = {a.seed}\n")
    f.write(f"* expected in-band (0..{a.bw:g} Hz) noise power = {inband:.4e} V^2\n")
    f.write(f"* series source between {a.np_} (+) and {a.nn} (-): V({a.np_}) = V({a.nn}) + noise\n")
    f.write(f"{a.name} {a.np_} {a.nn} pwl(\n")
    per_line = 6
    for i in range(0, a.n, per_line):
        pairs = " ".join(f"{(i + k) * T:.9e} {v[i + k]:.6e}" for k in range(min(per_line, a.n - i)))
        f.write(f"+ {pairs}\n")
    f.write("+ )\n")

print(f"wrote {a.out}: {a.n} points, t = 0 .. {(a.n - 1) * T:.6e} s, sigma = {sigma * 1e6:.2f} uV rms")
if a.snr is not None:
    print(f"  (target in-band noise-only SNR {a.snr:g} dB for signal power {a.ps:g} V^2, bw {a.bw:g} Hz)")
