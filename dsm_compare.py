"""Overlay the FFT of two 1-bit streams (e.g. with / without transient noise) on one plot.

Usage: python dsm_compare.py fileA fileB [--labels "noise,no noise"] [dsm_fft.py options]

All dsm_fft.py options (--skip, --n, --fin, --thresh, --window, --bw, --nharm, ...) are applied
to BOTH files, so they are analysed identically. Use matching N and fin in the two runs.
Output: fft_compare.png and a metrics table (A, B, A-B) in the figure.
"""
import os
import runpy
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def analyse(path, extra):
    """Run dsm_fft.py's analysis on one file without plotting; return its raw result."""
    here = os.path.dirname(os.path.abspath(__file__))
    argv = sys.argv
    sys.argv = ["dsm_fft.py", path] + extra + ["--noplot", "1"]
    try:
        g = runpy.run_path(os.path.join(here, "dsm_fft.py"), run_name="__main__")
    finally:
        sys.argv = argv
    return g["raw"], g


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    fa, fb = argv[0], argv[1]
    rest, labels, i = [], ["A", "B"], 2
    while i < len(argv):
        if argv[i] == "--labels":
            labels = [t.strip() for t in argv[i + 1].split(",")][:2]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    # positional fs / OSR / order of dsm_fft.py keep their defaults unless given as options
    print(f"=== A: {labels[0]}  ({fa})")
    ra, ga = analyse(fa, rest)
    print(f"=== B: {labels[1]}  ({fb})")
    rb, gb = analyse(fb, rest)
    if ra["N"] != rb["N"] or abs(ra["fs"] - rb["fs"]) > 1e-6:
        print("warning: the two records differ in N or fs; bins are not aligned")
    if abs(ra["fin"] - rb["fin"]) > 2 * ra["df"]:
        print("warning: fundamental differs between A and B -- compare with the same fin")

    bw = ra["bw"]
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.3, 1.25], height_ratios=[1.1, 1.1, 0.8])
    a1, a2, a3 = (fig.add_subplot(gs[k, 0]) for k in range(3))
    side = fig.add_subplot(gs[:, 1])
    side.axis("off")
    cols = ("C0", "C1")
    for r, lab, c in ((ra, labels[0], cols[0]), (rb, labels[1], cols[1])):
        a1.semilogx(r["f"][1:], r["d"][1:], lw=0.6, c=c, alpha=0.85, label=lab)
    a1.axvline(bw, c="r", ls="--", lw=1)
    a1.set(xlabel="Frequency [Hz]", ylabel="dBFS", ylim=(-140, 0),
           title=f"FFT overlay  (N = {ra['N']}, fin = {ra['fin']:.1f} Hz, window = {ra['win']})")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(loc="lower right", fontsize=9)

    panels = {a1: [(labels[0], ra["f"][1:], ra["d"][1:]), (labels[1], rb["f"][1:], rb["d"][1:])]}
    m = ra["f"] <= bw * 1.2
    for r, lab, c in ((ra, labels[0], cols[0]), (rb, labels[1], cols[1])):
        mm = r["f"] <= bw * 1.2
        a2.plot(r["f"][mm] / 1e3, r["d"][mm], "o-", ms=3, lw=0.8, c=c, label=lab)
    a2.axvline(bw / 1e3, c="r", ls="--", lw=1)
    a2.set(xlabel="kHz", ylabel="dBFS", ylim=(-140, 0), title="In-band")
    a2.grid(alpha=0.3)
    a2.legend(loc="lower left", fontsize=9)
    panels[a2] = [(labels[0], ra["f"][m] / 1e3, ra["d"][m]), (labels[1], rb["f"][m] / 1e3, rb["d"][m])]

    n = min(len(ra["d"]), len(rb["d"]))
    mm = ra["f"][:n] <= bw * 1.2
    diff = ra["d"][:n] - rb["d"][:n]
    a3.plot(ra["f"][:n][mm] / 1e3, diff[mm], "k.-", ms=3, lw=0.6)
    a3.axhline(0, c="0.5", lw=0.8)
    a3.axvline(bw / 1e3, c="r", ls="--", lw=1)
    a3.set(xlabel="kHz", ylabel="A - B [dB]", title=f"Difference ({labels[0]} minus {labels[1]})")
    a3.grid(alpha=0.3)
    panels[a3] = [("A-B", ra["f"][:n][mm] / 1e3, diff[mm])]

    rows = [("SINAD [dB]", "sinad", "{:.2f}"), ("SNR excl. harm. [dB]", "snr_nh", "{:.2f}"),
            ("THD [dB]", "thd", "{:.1f}"), ("ENOB [bit]", "enob", "{:.2f}"),
            ("SFDR [dB]", "sfdr", "{:.1f}"), ("fin [Hz]", "fin", "{:.1f}"),
            ("amplitude [dBFS]", "amp", "{:.2f}"), ("noise floor [dBFS/Hz]", "nd", "{:.1f}")]
    cells = [[fmt.format(ra[k]), fmt.format(rb[k]), f"{ra[k] - rb[k]:+.2f}"] for _, k, fmt in rows]
    ha, hb = dict((h, lv) for h, _, lv in ra["harm"]), dict((h, lv) for h, _, lv in rb["harm"])
    for h in sorted(set(ha) | set(hb)):
        if h in ha and h in hb and max(ha[h], hb[h]) > -105:
            rows.append((f"HD{h} [dBFS]", None, None))
            cells.append([f"{ha[h]:.1f}", f"{hb[h]:.1f}", f"{ha[h] - hb[h]:+.1f}"])
    tab = side.table(cellText=cells, rowLabels=[r[0] for r in rows],
                     colLabels=[labels[0], labels[1], "A - B"], cellLoc="center",
                     loc="upper center", bbox=[0.5, 0.45, 0.5, 0.5])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    side.text(0.0, 0.40, f"fs = {ra['fs'] / 1e6:g} MHz   bin = {ra['df']:.1f} Hz\n"
              f"BW = {bw / 1e3:.2f} kHz   window = {ra['win']}\n\n"
              "SINAD = signal / (noise + distortion)\n"
              "SNR   = signal / noise only (harmonics\n"
              "        removed, noise floor put back)\n"
              "noise floor: median bin, +-3 dB only;\n"
              "compare SINAD / SNR / THD instead.",
              transform=side.transAxes, ha="left", va="top", fontsize=8, family="monospace")
    plt.tight_layout()
    out = "fft_compare.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")
    if matplotlib.get_backend().lower() != "agg":
        CursorTool = ga["CursorTool"]
        cursor = CursorTool(fig, panels)          # noqa: F841  (keep a reference alive)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
