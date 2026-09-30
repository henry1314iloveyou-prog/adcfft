"""FFT analysis of a 1-bit delta-sigma ADC stream.

Usage: python dsm_fft.py [capture.txt] [fs] [OSR]
Needs: numpy, matplotlib   (pip install numpy matplotlib)
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "adc1khz1150927.txt"
fs = float(sys.argv[2]) if len(sys.argv) > 2 else 12e6
osr = float(sys.argv[3]) if len(sys.argv) > 3 else 256
bw = fs / (2 * osr)                       # signal bandwidth

x = np.loadtxt(path)                      # one 0/1 (or -1/+1) value per line
v = 2 * x - 1 if set(np.unique(x)) <= {0, 1} else x.astype(float)
v = v - v.mean()                          # remove DC
N = len(v)

# Blackman-Harris window
n = np.arange(N)
w = (0.35875 - 0.48829 * np.cos(2 * np.pi * n / N)
     + 0.14128 * np.cos(4 * np.pi * n / N) - 0.01168 * np.cos(6 * np.pi * n / N))

# One-sided power spectrum, 0 dB = full-scale sine (amplitude 1)
P = np.abs(np.fft.rfft(v * w)) ** 2
P[1:-1] *= 2
P = P / w.sum() ** 2 / 2
f = np.arange(len(P)) * fs / N
df = fs / N
bwb = int(bw / df)

# Fundamental: zero-padded FFT for a finer frequency estimate
Z = np.abs(np.fft.rfft(v * w, N * 32))
fz = np.arange(len(Z)) * fs / (N * 32)
sel = (fz > 3 * df) & (fz < bw)
fin = fz[sel][np.argmax(Z[sel])]

hw = 4                                    # Blackman-Harris main lobe: +/-4 bins
k0 = int(round(fin / df))
sig = np.zeros(len(P), bool)
sig[max(k0 - hw, 0):k0 + hw + 1] = True
mask = np.zeros(len(P), bool)             # in-band noise bins (skip DC)
mask[2:bwb + 1] = True
mask &= ~sig

Ps = P[sig].sum()
snr = 10 * np.log10(Ps / P[mask].sum())
sfdr = 10 * np.log10(Ps / P[mask].max())
enob = (snr - 1.76) / 6.02
amp = 10 * np.log10(2 * Ps)

print(f"N={N}  fs={fs:g} Hz  BW={bw:g} Hz  OSR={osr:g}  bin width={df:.1f} Hz")
print(f"fin  = {fin:.1f} Hz  ({amp:.2f} dBFS)")
print(f"SNR  = {snr:.2f} dB   ENOB = {enob:.2f} bit   SFDR = {sfdr:.1f} dB")

d = 10 * np.log10(2 * P + 1e-30)          # dBFS
fig, ax = plt.subplots(2, 1, figsize=(9, 8))
ax[0].semilogx(f[1:], d[1:], lw=0.6)
ax[0].axvline(bw, c="r", ls="--")
ax[0].set(xlabel="Frequency [Hz]", ylabel="dBFS", ylim=(-140, 0),
          title=f"1-bit stream, N={N}, fs={fs/1e6:g} MHz")
ax[0].grid(True, which="both", alpha=0.3)
m = f <= bw * 1.2
ax[1].plot(f[m] / 1e3, d[m], lw=0.8)
ax[1].axvline(bw / 1e3, c="r", ls="--")
ax[1].set(xlabel="kHz", ylabel="dBFS", ylim=(-140, 0),
          title=f"In-band: fin={fin:.0f} Hz, SNR={snr:.1f} dB, ENOB={enob:.1f}")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fft_result.png", dpi=110)
if matplotlib.get_backend().lower() != "agg":
    plt.show()
