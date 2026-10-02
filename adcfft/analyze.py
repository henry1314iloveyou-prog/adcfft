"""Spectral test of a 1-bit delta-sigma ADC stream (port of dsm_analyze.m)."""
import numpy as np


def load_bits(path):
    """Read a one-value-per-line capture, skipping non-numeric header lines."""
    vals = []
    with open(path) as fh:
        for line in fh:
            try:
                vals.append(float(line.split()[0]))
            except (ValueError, IndexError):
                continue
    return np.asarray(vals)


def _window(name, n):
    k = np.arange(n)
    name = name.lower()
    if name == "rect":
        return np.ones(n), 1
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / n), 3
    return (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n)
            + 0.14128 * np.cos(4 * np.pi * k / n)
            - 0.01168 * np.cos(6 * np.pi * k / n)), 5


def analyze(bits, fs, bw, window="blackmanharris", nharm=5, dc_bins=3,
            full_scale=1.0, verbose=True):
    """Return dict with fin, Asig_dBFS, SNR, SINAD, SFDR, THD, ENOB, OSR, f, PdB."""
    x = np.asarray(bits, dtype=float).ravel()
    if np.all((x == 0) | (x == 1)):
        x = 2 * x - 1
    x = x / full_scale
    n = x.size
    x = x - x.mean()

    w, hw = _window(window, n)
    X = np.fft.fft(x * w)
    nb = n // 2 + 1
    P = np.abs(X[:nb]) ** 2
    P[1:-1] *= 2
    P = P / w.sum() ** 2 / 2
    f = np.arange(nb) * fs / n
    bw_bin = min(nb, int(bw // (fs / n)) + 1)      # 1-based last in-band bin

    # 0-based helpers; MATLAB 1-based index i  <->  python i-1
    lo = dc_bins + 1                                # first in-band bin (0-based)
    hi = bw_bin                                     # exclusive upper bound
    k0 = lo + int(np.argmax(P[lo:hi]))

    def sig_idx(kc):
        return np.arange(max(0, kc - hw), min(nb, kc + hw + 1))

    psig = P[sig_idx(k0)].sum()

    h_idx = set()
    ph = 0.0
    for h in range(2, nharm + 2):
        kh = (k0 * h) % n
        if kh > n / 2:
            kh = n - kh
        if kh < hi and kh > k0 + 2 * hw:
            idx = sig_idx(kh)
            h_idx.update(idx.tolist())
            ph += P[idx].sum()

    mask = np.zeros(nb, dtype=bool)
    mask[lo:hi] = True
    mask[sig_idx(k0)] = False
    harm = np.zeros(nb, dtype=bool)
    harm[list(h_idx)] = True
    pn_noharm = P[mask & ~harm].sum()
    pn_all = P[mask].sum()

    snr = 10 * np.log10(psig / pn_noharm)
    sinad = 10 * np.log10(psig / pn_all)
    thd = 10 * np.log10(max(ph, np.finfo(float).eps) / psig)
    sfdr = 10 * np.log10(psig / P[mask].max())
    enob = (sinad - 1.76) / 6.02

    r = dict(fin=f[k0], Asig_dBFS=10 * np.log10(2 * psig), SNR=snr, SINAD=sinad,
             SFDR=sfdr, THD=thd, ENOB=enob, OSR=fs / (2 * bw), f=f,
             PdB=10 * np.log10(P + np.finfo(float).eps))
    if verbose:
        print(f"N = {n}, fs = {fs:g} Hz, BW = {bw:g} Hz, OSR = {r['OSR']:g}")
        print(f"Fundamental : {r['fin']:.4f} Hz  ({r['Asig_dBFS']:.2f} dBFS)")
        print(f"SNR         : {snr:.2f} dB")
        print(f"SINAD       : {sinad:.2f} dB")
        print(f"THD         : {thd:.2f} dB")
        print(f"SFDR        : {sfdr:.2f} dB")
        print(f"ENOB        : {enob:.2f} bits")
    return r


def plot(r, bw, path=None):
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.semilogx(r["f"][1:], r["PdB"][1:] + 10 * np.log10(2))
    ax.axvline(bw, color="r", ls="--", label="BW")
    ax.grid(True)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power [dB, rel. full-scale sine]")
    ax.set_title(f"SNR {r['SNR']:.1f} dB, SINAD {r['SINAD']:.1f} dB, ENOB {r['ENOB']:.2f}")
    ax.legend()
    if path:
        fig.savefig(path, dpi=120)
    else:
        plt.show()
