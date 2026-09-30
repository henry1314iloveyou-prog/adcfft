"""FFT analysis of a 1-bit delta-sigma ADC stream, raw and after a CIC decimator.

Usage: python dsm_fft.py [capture.txt] [fs] [OSR] [cic_order]
Needs: numpy, matplotlib   (pip install numpy matplotlib)

The CIC matches deci_hspice.mdl: `order` integrators at fs, decimate by R=OSR,
`order` combs at fs/R, unity DC gain.
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "adc1khz1150927.txt"
fs = float(sys.argv[2]) if len(sys.argv) > 2 else 12e6
osr = int(float(sys.argv[3])) if len(sys.argv) > 3 else 256
order = int(sys.argv[4]) if len(sys.argv) > 4 else 3
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


def analyze(v, fs, bw, name):
    """Return dict of spectrum + metrics. 0 dB = full-scale sine (amplitude 1)."""
    v = v - v.mean()                      # remove DC
    N = len(v)
    n = np.arange(N)
    w = (0.35875 - 0.48829 * np.cos(2 * np.pi * n / N)
         + 0.14128 * np.cos(4 * np.pi * n / N) - 0.01168 * np.cos(6 * np.pi * n / N))
    P = np.abs(np.fft.rfft(v * w)) ** 2
    P[1:-1] *= 2
    P = P / w.sum() ** 2 / 2
    df = fs / N
    f = np.arange(len(P)) * df
    bwb = min(int(bw / df), len(P) - 1)

    Z = np.abs(np.fft.rfft(v * w, N * 32))          # fine fin estimate
    fz = np.arange(len(Z)) * fs / (N * 32)
    sel = (fz > 3 * df) & (fz < bw)
    fin = fz[sel][np.argmax(Z[sel])]

    hw = 4                                # Blackman-Harris main lobe +/-4 bins
    k0 = int(round(fin / df))
    sig = np.zeros(len(P), bool)
    sig[max(k0 - hw, 0):k0 + hw + 1] = True
    mask = np.zeros(len(P), bool)
    mask[2:bwb + 1] = True
    mask &= ~sig
    Ps = P[sig].sum()
    snr = 10 * np.log10(Ps / P[mask].sum())
    r = dict(name=name, N=N, fs=fs, f=f, d=10 * np.log10(2 * P + 1e-30), fin=fin,
             amp=10 * np.log10(2 * Ps), snr=snr, enob=(snr - 1.76) / 6.02,
             sfdr=10 * np.log10(Ps / P[mask].max()), df=df)
    print(f"[{name}] N={N} fs={fs:g} Hz bin={df:.1f} Hz | fin={fin:.1f} Hz "
          f"({r['amp']:.2f} dBFS) SNR={snr:.2f} dB ENOB={r['enob']:.2f} "
          f"SFDR={r['sfdr']:.1f} dB")
    return r


x = np.loadtxt(path)                      # one 0/1 (or -1/+1) value per line
v = 2 * x - 1 if set(np.unique(x)) <= {0, 1} else x.astype(float)

raw = analyze(v, fs, bw, "raw 1-bit")
y = cic_decimate(v, osr, order)[order + 2:]          # drop CIC start-up transient
dec = analyze(y, fs / osr, bw, f"CIC{order} /{osr}")

fig, ax = plt.subplots(3, 1, figsize=(9, 11))
for r, a in ((raw, ax[0]), (dec, ax[1])):
    a.semilogx(r["f"][1:], r["d"][1:], lw=0.7)
    a.axvline(bw, c="r", ls="--")
    a.set(xlabel="Frequency [Hz]", ylabel="dBFS", ylim=(-140, 0),
          title=f"{r['name']}: fin={r['fin']:.0f} Hz, SNR={r['snr']:.1f} dB, "
                f"ENOB={r['enob']:.1f}")
    a.grid(True, which="both", alpha=0.3)
m = raw["f"] <= bw * 1.2
ax[2].plot(raw["f"][m] / 1e3, raw["d"][m], lw=0.8, label="raw")
m = dec["f"] <= bw * 1.2
ax[2].plot(dec["f"][m] / 1e3, dec["d"][m], "o-", ms=3, lw=0.8, label="after CIC")
ax[2].axvline(bw / 1e3, c="r", ls="--")
ax[2].set(xlabel="kHz", ylabel="dBFS", ylim=(-140, 0), title="In-band comparison")
ax[2].legend()
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fft_result.png", dpi=110)
if matplotlib.get_backend().lower() != "agg":
    plt.show()
