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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("file", nargs="?", default="meannoise.txt")
    p.add_argument("--fs", type=float, default=6144000 * 2)
    p.add_argument("--osr", type=int, default=256)
    p.add_argument("--vth", type=float, default=0.9, help="1-bit slicing threshold [V]")
    p.add_argument("--lsb-fs", type=float, default=32768, help="LSBs per full scale (+/-1)")
    p.add_argument("--vfsd", type=float, default=1.5, help="differential full scale [V]")
    p.add_argument("--skip", type=int, default=5, help="decimated start-up samples to drop")
    p.add_argument("--out", default="noise_lsb.png")
    a = p.parse_args()

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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(code, lw=0.7)
    ax.set(xlabel="output sample", ylabel="output code [LSB]",
           ylim=(-a.lsb_fs, a.lsb_fs), title=f"mean = {z:.1f} LSB, std = {sd:.2f} LSB (FS = +/-{a.lsb_fs:g})")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
