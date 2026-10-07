"""Overlay the FFT of several 1-bit streams on one plot (with / without noise, injected-noise runs, ...).

Usage: python dsm_compare.py fileA fileB [fileC ...] [fs OSR] [options] [dsm_fft.py options]
  --labels "a,b,c"   one label per file
  --ref K            index (0-based, negative allowed) of the reference run; differences are taken
                     against it (default: the last file)
  --inject "70,80,90,-"   noise-only SNR [dB] injected into each run ('-' = none, e.g. the baseline);
                     the table then shows the EXPECTED total SNR
                     = -10*log10(10^(-SNR_ref/10) + 10^(-inject/10)) next to the measured one

All other options (--skip, --n, --fin, --thresh, --window, --bw, --nharm, ...) go to dsm_fft.py and are
applied to EVERY file, so the runs are analysed identically (use the same N and fin).
Output: fft_compare.png and a metrics table.
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


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main(argv):
    files = []
    while argv and not argv[0].startswith("--") and not _isnum(argv[0]):
        files.append(argv.pop(0))
    if len(files) < 2:
        sys.exit(__doc__)
    labels, ref, inject, rest, i = None, -1, None, [], 0
    while i < len(argv):
        if argv[i] == "--labels":
            labels = [t.strip() for t in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--ref":
            ref = int(argv[i + 1])
            i += 2
        elif argv[i] == "--inject":
            inject = [None if t.strip() in ("-", "", "none") else float(t) for t in argv[i + 1].split(",")]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    k = len(files)
    labels = (labels or [chr(65 + j) for j in range(k)])
    if len(labels) != k:
        sys.exit(f"--labels needs {k} entries")
    if inject is not None and len(inject) != k:
        sys.exit(f"--inject needs {k} entries")
    ref %= k

    res, gs = [], []
    for f, lab in zip(files, labels):
        print(f"=== {lab}  ({f})")
        r, g = analyse(f, rest)
        res.append(r)
        gs.append(g)
    r0 = res[ref]
    for lab, r in zip(labels, res):
        if r["N"] != r0["N"] or abs(r["fs"] - r0["fs"]) > 1e-6:
            print(f"warning: {lab} differs from the reference in N or fs; bins are not aligned")
        if abs(r["fin"] - r0["fin"]) > 2 * r0["df"]:
            print(f"warning: {lab}: fundamental differs from the reference -- use the same fin")

    bw = r0["bw"]
    colors = [f"C{j}" for j in range(k)]
    fig = plt.figure(figsize=(15, 11.5))
    gs_ = fig.add_gridspec(3, 2, width_ratios=[3.0, 1.5], height_ratios=[1.1, 1.1, 0.8])
    a1, a2, a3 = (fig.add_subplot(gs_[j, 0]) for j in range(3))
    side = fig.add_subplot(gs_[:, 1])
    side.axis("off")
    panels = {a1: [], a2: [], a3: []}
    for j, (r, lab) in enumerate(zip(res, labels)):
        lw, al = (0.9, 1.0) if j == ref else (0.6, 0.8)
        a1.semilogx(r["f"][1:], r["d"][1:], lw=lw, c=colors[j], alpha=al, label=lab)
        panels[a1].append((lab, r["f"][1:], r["d"][1:]))
        m = r["f"] <= bw * 1.2
        a2.plot(r["f"][m] / 1e3, r["d"][m], "o-", ms=3, lw=0.8, c=colors[j], label=lab)
        panels[a2].append((lab, r["f"][m] / 1e3, r["d"][m]))
    a1.axvline(bw, c="r", ls="--", lw=1)
    a1.set(xlabel="Frequency [Hz]", ylabel="dBFS", ylim=(-140, 0),
           title=f"FFT overlay  (N = {r0['N']}, fin = {r0['fin']:.1f} Hz, window = {r0['win']})")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(loc="lower right", fontsize=8)
    a2.axvline(bw / 1e3, c="r", ls="--", lw=1)
    a2.set(xlabel="kHz", ylabel="dBFS", ylim=(-140, 0), title="In-band")
    a2.grid(alpha=0.3)
    a2.legend(loc="lower left", fontsize=8)

    n = min(len(r["d"]) for r in res)
    mm = r0["f"][:n] <= bw * 1.2
    for j, (r, lab) in enumerate(zip(res, labels)):
        if j == ref:
            continue
        diff = r["d"][:n] - r0["d"][:n]
        a3.plot(r0["f"][:n][mm] / 1e3, diff[mm], ".-", ms=3, lw=0.6, c=colors[j], label=lab)
        panels[a3].append((lab, r0["f"][:n][mm] / 1e3, diff[mm]))
    a3.axhline(0, c="0.5", lw=0.8)
    a3.axvline(bw / 1e3, c="r", ls="--", lw=1)
    a3.set(xlabel="kHz", ylabel="run - ref [dB]", title=f"Difference to the reference: {labels[ref]}")
    a3.grid(alpha=0.3)
    a3.legend(loc="upper right", fontsize=7)

    # --- table: values per run ---
    rows = [("SINAD [dB]", "sinad", "{:.2f}"), ("SNR excl. harm. [dB]", "snr_nh", "{:.2f}"),
            ("THD [dB]", "thd", "{:.1f}"), ("ENOB [bit]", "enob", "{:.2f}"),
            ("SFDR [dB]", "sfdr", "{:.1f}"), ("fin [Hz]", "fin", "{:.1f}"),
            ("amplitude [dBFS]", "amp", "{:.2f}")]
    def f(v, fmt):
        return fmt.format(v) if v == v else "n/a"
    cells = [[f(r[key], fmt) for r in res] for _, key, fmt in rows]
    names = [lab for lab, _, _ in rows]
    if inject is not None:
        base, base_d = r0["snr_nh"], r0["sinad"]
        def add(b, inj):
            return -10 * np.log10(10 ** (-b / 10) + 10 ** (-inj / 10))
        exp = ["-" if inj is None else f"{add(base, inj):.2f}" for inj in inject]
        exp_d = ["-" if inj is None else f"{add(base_d, inj):.2f}" for inj in inject]
        cells.insert(2, [("-" if inj is None else f"{inj:g}") for inj in inject])
        names.insert(2, "injected SNR [dB]")
        cells.insert(3, exp)
        names.insert(3, "expected SNR [dB]")
        cells.insert(4, exp_d)
        names.insert(4, "expected SINAD [dB]")
    hs = [dict((h, lv) for h, _, lv in r["harm"]) for r in res]
    for h in sorted(set.intersection(*[set(d) for d in hs])):
        if max(d[h] for d in hs) > -105:
            names.append(f"HD{h} [dBFS]")
            cells.append([f"{d[h]:.1f}" for d in hs])
    t1 = side.table(cellText=cells, rowLabels=names, colLabels=labels, cellLoc="center",
                    loc="upper right", bbox=[0.38, 0.52, 0.62, 0.46])
    t1.auto_set_font_size(False)
    t1.set_fontsize(7)

    # --- table: difference to the reference ---
    others = [j for j in range(k) if j != ref]
    drows = [("dSINAD [dB]", "sinad"), ("dSNR excl. harm. [dB]", "snr_nh"), ("dTHD [dB]", "thd"),
             ("dSFDR [dB]", "sfdr")]
    dcells = [[(f"{res[j][key] - r0[key]:+.2f}" if (res[j][key] == res[j][key] and r0[key] == r0[key]) else "n/a")
               for j in others] for _, key in drows]
    t2 = side.table(cellText=dcells, rowLabels=[d[0] for d in drows],
                    colLabels=[labels[j] for j in others], cellLoc="center", loc="upper right",
                    bbox=[0.38, 0.27, 0.62, 0.20])
    t2.auto_set_font_size(False)
    t2.set_fontsize(7)
    side.text(0.38, 0.49, f"vs. reference '{labels[ref]}'", transform=side.transAxes, fontsize=7.5,
              ha="left", va="bottom")
    note = (f"fs = {r0['fs'] / 1e6:g} MHz   bin = {r0['df']:.1f} Hz\nBW = {bw / 1e3:.2f} kHz   "
            f"window = {r0['win']}\n\nSINAD = signal / (noise + distortion)\n"
            "SNR   = signal / noise only (harmonics\n        removed, noise floor put back)\n"
            "expected SNR = baseline noise + injected\n  noise, added in power. A single run is\n"
            "  only good to ~1 dB (1 sigma).")
    side.text(0.0, 0.22, note, transform=side.transAxes, ha="left", va="top", fontsize=7.5,
              family="monospace")
    plt.tight_layout()
    out = "fft_compare.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")
    if inject is not None:
        print("\nmeasured vs expected (baseline noise + injected noise, added in power):")
        for lab, r, inj, e, ed in zip(labels, res, inject, exp, exp_d):
            if inj is not None:
                print(f"  {lab:20s} SNR  excl. harm.: expected {e:>6s}  measured {r['snr_nh']:6.2f}  "
                      f"diff {r['snr_nh'] - float(e):+5.2f} dB   |   SINAD: expected {ed:>6s}  "
                      f"measured {r['sinad']:6.2f}  diff {r['sinad'] - float(ed):+5.2f} dB")
    if matplotlib.get_backend().lower() != "agg":
        CursorTool = gs[0]["CursorTool"]
        cursor = CursorTool(fig, panels)          # noqa: F841  (keep a reference alive)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
