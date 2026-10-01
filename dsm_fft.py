"""FFT analysis of a 1-bit delta-sigma ADC stream, raw and after a CIC decimator.

Usage: python dsm_fft.py [capture.txt] [fs] [OSR] [cic_order] [--skip S] [--n N] [--fin F]
                                  [--window bh|hann|rect] [--nocic 1]
  --skip S  drop the first S samples (start-up transient), default 0
  --n N     use N contiguous samples after the skip (default: all, cut to a
            multiple of OSR so the CIC output is whole)
  --fin F   expected input frequency [Hz]; checks coherence (M=fin*N/fs integer,
            gcd(M,N)=1). Coherent example: fs=12 MHz, N=65536, M=11 -> 2014.16015625 Hz
  --window  bh (Blackman-Harris, default, signal = +/-4 bins), hann (+/-2 bins),
            rect (signal = 1 bin; only for coherent sampling, noise-shaped
            high-freq noise may leak in-band)
  --nocic 1 skip the CIC stage (use when the record is short)
Needs: numpy, matplotlib   (pip install numpy matplotlib)

The CIC matches deci_hspice.mdl: `order` integrators at fs, decimate by R=OSR,
`order` combs at fs/R, unity DC gain.
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import math

WINDOWS = {"bh": 4, "hann": 2, "rect": 0}         # signal half-width [bins]

args, opts, it = [], {}, iter(sys.argv[1:])
for a in it:
    if a.startswith("--"):
        opts[a[2:]] = next(it)
    else:
        args.append(a)
path = args[0] if len(args) > 0 else "adc1khz1150927.txt"
fs = float(args[1]) if len(args) > 1 else 12e6
osr = int(float(args[2])) if len(args) > 2 else 256
order = int(args[3]) if len(args) > 3 else 3
skip = int(float(opts.get("skip", 0)))
window = opts.get("window", "bh")
if window not in WINDOWS:
    sys.exit("--window must be bh, hann or rect")
use_cic = opts.get("nocic", "0") in ("0", "false")
bw = fs / (2 * osr)                       # signal bandwidth


def cic_decimate(x, R, order):
    """CIC decimator (integrators -> downsample by R -> combs), unity DC gain."""
    y = x.astype(np.float64)
    for _ in range(order):
        y = np.cumsum(y)                  # integrators at fs
    y = y[R - 1::R]                       # decimate
    for _ in range(order):
        y = np.diff(y, prepend=0.0)       # combs at fs/R
    return y / R ** order


def make_window(kind, N):
    n = np.arange(N)
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)
    if kind == "rect":
        return np.ones(N)
    return (0.35875 - 0.48829 * np.cos(2 * np.pi * n / N)
            + 0.14128 * np.cos(4 * np.pi * n / N) - 0.01168 * np.cos(6 * np.pi * n / N))


def analyze(v, fs, bw, name, fin_expect=None):
    """Return dict of spectrum + metrics. 0 dB = full-scale sine (amplitude 1)."""
    v = v - v.mean()                      # remove DC
    N = len(v)
    w = make_window(window, N)
    P = np.abs(np.fft.rfft(v * w)) ** 2
    P[1:-1] *= 2
    P = P / w.sum() ** 2 / 2
    df = fs / N
    f = np.arange(len(P)) * df
    bwb = min(int(bw / df), len(P) - 1)

    Z = np.abs(np.fft.rfft(v * w, N * 32))          # fine fin estimate
    fz = np.arange(len(Z)) * fs / (N * 32)
    sel = (fz > 3 * df) & (fz < bw)
    fin_peak = fz[sel][np.argmax(Z[sel])]             # strongest in-band peak
    if fin_expect is None:
        fin = fin_peak
    else:                                             # lock onto the expected tone
        near = (fz > fin_expect - 1 * df) & (fz < fin_expect + 1 * df)
        fin = fz[near][np.argmax(Z[near])]
        d_near = 20 * np.log10(Z[near].max() / (w.sum() / 2) + 1e-30)
        d_peak = 20 * np.log10(Z[sel].max() / (w.sum() / 2) + 1e-30)
        print(f"[{name}] tone near expected {fin_expect:.1f} Hz: {d_near:.1f} dBFS; "
              f"strongest in-band peak: {fin_peak:.1f} Hz at {d_peak:.1f} dBFS")
        if d_peak - d_near > 6:
            print(f"[{name}] WARNING: expected tone is >6 dB below the strongest peak "
                  f"-- input tone missing or wrong frequency/fs; SNR below is NOT valid")

    hw = WINDOWS[window]
    k0 = int(round(fin / df))
    sig = np.zeros(len(P), bool)
    sig[max(k0 - hw, 0):k0 + hw + 1] = True
    mask = np.zeros(len(P), bool)
    mask[2:bwb + 1] = True
    mask &= ~sig
    Ps = P[sig].sum()
    enbw = N * np.sum(w ** 2) / np.sum(w) ** 2     # window noise bandwidth [bins]
    snr = 10 * np.log10(Ps / P[mask].sum())
    r = dict(name=name, N=N, fs=fs, f=f, d=10 * np.log10(4 * P + 1e-30), fin=fin,
             amp=10 * np.log10(4 * Ps / enbw), snr=snr, enob=(snr - 1.76) / 6.02,
             sfdr=10 * np.log10(Ps / P[mask].max()), df=df)
    print(f"[{name}] N={N} fs={fs:g} Hz bin={df:.1f} Hz | fin={fin:.1f} Hz "
          f"({r['amp']:.2f} dBFS) SNR={snr:.2f} dB ENOB={r['enob']:.2f} "
          f"SFDR={r['sfdr']:.1f} dB")
    return r


try:
    x = np.loadtxt(path)                  # one 0/1 (or -1/+1) value per line
except ValueError:                        # header line (e.g. exported from WaveView)
    x = np.loadtxt(path, skiprows=1)
if x.ndim == 2:                           # two columns: time, value
    t, x = x[:, 0], x[:, 1]
    dt = np.diff(t)
    print(f"time column: median step {np.median(dt):.4g} s "
          f"(expected {1 / fs:.4g} s), min {dt.min():.4g}, max {dt.max():.4g}")
    if abs(np.median(dt) * fs - 1) > 1e-3 or dt.max() > 1.5 / fs:
        sys.exit("ERROR: time step is not 1/fs -- the file is not one value per clock "
                 "(WaveView may have exported only the transitions); resample first")
levels = np.unique(x)
runs = np.diff(np.flatnonzero(np.diff(x) != 0), prepend=-1)
print(f"file: {len(x)} samples, levels {levels[:4]}{'...' if len(levels) > 4 else ''}, "
      f"fraction of 1s {np.mean(x > x.min() + (x.max() - x.min()) / 2):.3f}, "
      f"longest run {runs.max() if len(runs) else len(x)}")
if len(levels) > 2:
    sys.exit("ERROR: more than two levels -- this is a voltage waveform; "
             "threshold it (qout > 0.9 V) to 0/1 first")
v = 2 * x - 1 if set(levels) <= {0, 1} else x.astype(float)

v = v[skip:]
n_use = int(float(opts["n"])) if "n" in opts else len(v) // osr * osr
if n_use > len(v):
    sys.exit(f"need {n_use} samples after skip={skip}, file has only {len(v)}")
v = v[:n_use]
if n_use % osr:
    print(f"warning: N={n_use} is not a multiple of OSR={osr}")
if "fin" in opts:
    M = float(opts["fin"]) * n_use / fs
    ok = abs(M - round(M)) < 1e-6 and math.gcd(int(round(M)), n_use) == 1
    print(f"coherence: M = fin*N/fs = {M:.6f} -> {'OK' if ok else 'NOT coherent (M must be an integer coprime to N)'}")

fin_exp = float(opts["fin"]) if "fin" in opts else None
raw = analyze(v, fs, bw, "raw 1-bit", fin_exp)
dec = None
if use_cic:
    y = cic_decimate(v, osr, order)[order + 2:]      # drop CIC start-up transient
    dec = analyze(y, fs / osr, bw, f"CIC{order} /{osr}", fin_exp)

rs = [raw] + ([dec] if dec else [])
fig, ax = plt.subplots(len(rs) + 1, 1, figsize=(9, 3.7 * (len(rs) + 1)))
for r, a in zip(rs, ax):
    a.semilogx(r["f"][1:], r["d"][1:], lw=0.7)
    a.axvline(bw, c="r", ls="--")
    a.set(xlabel="Frequency [Hz]", ylabel="dBFS", ylim=(-140, 0),
          title=f"{r['name']}: fin={r['fin']:.0f} Hz, SNR={r['snr']:.1f} dB, "
                f"ENOB={r['enob']:.1f}  [{window}]")
    a.grid(True, which="both", alpha=0.3)
for r, lab in zip(rs, ("raw", "after CIC")):
    m = r["f"] <= bw * 1.2
    ax[-1].plot(r["f"][m] / 1e3, r["d"][m], "o-", ms=3, lw=0.8, label=lab)
ax[-1].axvline(bw / 1e3, c="r", ls="--")
ax[-1].set(xlabel="kHz", ylabel="dBFS", ylim=(-140, 0), title="In-band")
ax[-1].legend()
ax[-1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fft_result.png", dpi=110)
if matplotlib.get_backend().lower() != "agg":
    plt.show()
