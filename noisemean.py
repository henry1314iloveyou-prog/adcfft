#!/usr/bin/env python3
"""Python port of noisemean.m: noise (in LSB) of the 1-bit DSM ADC at fixed input.

  python noisemean.py [meannoise.txt] [--fs 12.288e6] [--osr 256] [--vth 0.9] [--lsb-fs 32768]

meannoise.txt : 1-bit modulator output (0 / 1.8 V), one sample per line, rate fs.
Decimation    : sinc3 (3 integrators @fs, decimate by OSR, 3 combs), same as deci_hspice.mdl.
Output code   : 16-bit signed, bit = +/-1 -> +/-32768 LSB (differential +/-VFSD = +/-1.5 V,
                Vcm = 0.75 V, so 1 LSB = VFSD/32768).
"""
import argparse
import numpy as np


def sinc3_decimate(bits, osr, order=3):
    """CIC (integrate @fs, decimate, comb) with exact integer arithmetic; DC gain normalised to 1."""
    x = np.asarray(bits, dtype=np.int64)
    for _ in range(order):
        x = np.cumsum(x)
    x = x[osr - 1::osr]
    for _ in range(order):
        x = np.diff(x, prepend=0)
    return x / float(osr ** order)


def spectrum(x, fs, dc_bins=3):
    """Hann-windowed one-sided power spectrum, scaled so a full-scale (amplitude 1) sine = 0 dBFS.
    Returns f, P (power, FS-sine = 0.5), and the window's noise-equivalent bandwidth in bins."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    amp = np.abs(X) / (w.sum() / 2)              # peak amplitude of a coherent tone
    P = amp ** 2 / 2                             # power (sine of amplitude A -> A^2/2)
    f = np.fft.rfftfreq(n, 1 / fs)
    enbw = n * np.sum(w ** 2) / w.sum() ** 2
    return f, P, enbw


def snr_from_spectrum(f, P, enbw, dc_bins=3, tone_db=15.0, tone_halfwidth=2):
    """If a tone stands >= tone_db above the median noise bin, SNR = tone power / rest.
    Otherwise (DC input) SNR is quoted against a full-scale sine (power 0.5)."""
    Pn = P.copy()
    Pn[:dc_bins] = 0
    k = int(np.argmax(Pn))
    med = np.median(Pn[dc_bins:])
    has_tone = Pn[k] > med * 10 ** (tone_db / 10)
    if has_tone:
        lo, hi = max(k - tone_halfwidth, dc_bins), k + tone_halfwidth + 1
        sig = Pn[lo:hi].sum() / enbw
        noise = (Pn.sum() - Pn[lo:hi].sum())
        return has_tone, f[k], sig, noise
    return has_tone, None, 0.5, Pn.sum() / enbw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("file", nargs="?", default="meannoise.txt")
    p.add_argument("--fs", type=float, default=6144000 * 2)
    p.add_argument("--osr", type=int, default=256)
    p.add_argument("--vth", type=float, default=0.9, help="1-bit slicing threshold [V]")
    p.add_argument("--lsb-fs", type=float, default=32768, help="LSBs per full scale (+/-1)")
    p.add_argument("--vfsd", type=float, default=1.5, help="differential full scale [V]")
    p.add_argument("--skip", type=int, default=5, help="decimated start-up samples to drop")
    p.add_argument("--dc-bins", type=int, default=1, help="lowest FFT bins excluded from noise (default 1 = DC only)")
    p.add_argument("--out", default="noise_lsb", help="output basename (.png and .pdf added)")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="do not open the interactive matplotlib window (default: open)")
    p.add_argument("--open", metavar="PKL", help="open a saved .fig.pkl in an interactive matplotlib window")
    a, _ = p.parse_known_args()        # tolerant of IDE-injected argv
    if a.open:
        import pickle
        import matplotlib.pyplot as plt
        pickle.load(open(a.open, "rb"))
        plt.show()
        return

    import os
    if not os.path.exists(a.file):          # IDEs often run with a different cwd
        a.file = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.file)
    raw = np.loadtxt(a.file)
    bits = np.where(raw > a.vth, 1, -1)
    y = sinc3_decimate(bits, a.osr)
    code = np.round(y * a.lsb_fs)[a.skip:]          # 16-bit signed code
    fs4 = a.fs / a.osr

    z = code.mean()
    print(f"fs = {a.fs/1e6:g} MHz, OSR = {a.osr}, fout = {fs4/1e3:g} kHz, {code.size} output samples")
    print(f"mean   = {z:.2f} LSB  ({z / a.lsb_fs * a.vfsd * 1e3:.2f} mV differential)")
    print(f"max-mean = {code.max() - z:.2f} LSB, mean-min = {z - code.min():.2f} LSB, pk-pk = {np.ptp(code):.0f} LSB")
    sd = code.std(ddof=1)
    print(f"noise rms (std) = {sd:.3f} LSB = {sd / a.lsb_fs * a.vfsd * 1e6:.1f} uV (diff)")
    print(f"ENOB (noise only, full-scale sine) = {np.log2(2 * a.lsb_fs / (sd * np.sqrt(12))):.2f} bits")

    # ---- spectrum / SNR of the decimated output (full scale = 1.0) ----
    u = code / a.lsb_fs
    f, P, enbw = spectrum(u, fs4)
    tone, ftone, sig, noise = snr_from_spectrum(f, P, enbw, dc_bins=a.dc_bins)
    snr = 10 * np.log10(sig / noise)
    if tone:
        print(f"Tone found at {ftone / 1e3:.2f} kHz: SNR = {snr:.2f} dB (tone vs all other bins, DC excluded)")
    else:
        print(f"No tone (DC input): SNR vs full-scale sine = {snr:.2f} dB "
              f"(FS sine rms {a.lsb_fs / np.sqrt(2):.0f} LSB / noise rms {np.sqrt(noise) * a.lsb_fs:.2f} LSB, "
              f"first {a.dc_bins} bin(s) excluded)")
    print(f"ENOB from SNR = {(snr - 1.76) / 6.02:.2f} bits")

    # raw 1-bit stream spectrum: shows the noise shaping, and in-band noise as a cross-check
    fb, Pb, enbw_b = spectrum(bits, a.fs)
    bw = fs4 / 2
    inb = (fb > (a.dc_bins - 0.5) * a.fs / bits.size) & (fb <= bw)
    snr_b = 10 * np.log10(0.5 / (Pb[inb].sum() / enbw_b))
    print(f"1-bit stream, in-band 0..{bw / 1e3:g} kHz: SNR vs FS sine = {snr_b:.2f} dB (cross-check)")

    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 10))
    ax[0].plot(code, lw=0.7)
    ax[0].set(xlabel="output sample", ylabel="output code [LSB]",
              ylim=(-a.lsb_fs, a.lsb_fs), title=f"mean = {z:.1f} LSB, std = {sd:.2f} LSB (FS = +/-{a.lsb_fs:g})")
    dbf = lambda p: 10 * np.log10(np.maximum(p / 0.5, 1e-20))   # 0 dBFS = full-scale sine
    ax[1].plot(f / 1e3, dbf(P), lw=0.8)
    ax[1].set(xlabel="frequency [kHz]", ylabel="dBFS", ylim=(-160, 0),
              title=f"Decimated output spectrum (fs={fs4 / 1e3:g} kHz), SNR = {snr:.1f} dB")
    ax[2].semilogx(fb[1:] / 1e3, dbf(Pb[1:]), lw=0.5)
    ax[2].axvline(bw / 1e3, color="r", ls="--", lw=0.8, label=f"band edge {bw / 1e3:g} kHz")
    ax[2].set(xlabel="frequency [kHz]", ylabel="dBFS", ylim=(-160, 0),
              title=f"1-bit stream spectrum (fs={a.fs / 1e6:g} MHz), in-band SNR = {snr_b:.1f} dB")
    ax[2].legend(loc="lower right")
    for x in ax:
        x.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=120)
    fig.savefig(a.out + ".pdf")           # vector: zoom as far as you like in any PDF viewer
    import pickle
    with open(a.out + ".fig.pkl", "wb") as f:
        pickle.dump(fig, f)               # reopen: python noisemean.py --open noise_lsb.fig.pkl
    print(f"saved {a.out}.png, {a.out}.pdf, {a.out}.fig.pkl")
    if a.show:
        plt.show()                        # interactive window: mouse zoom / pan toolbar


if __name__ == "__main__":
    main()
