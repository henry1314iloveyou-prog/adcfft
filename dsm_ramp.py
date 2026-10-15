#!/usr/bin/env python3
"""Ramp test of a 1-bit delta-sigma ADC (gain / offset / INL / DNL / missing codes).

  python dsm_ramp.py sim    [--tr 4] [--fs 12e6]          behavioural 2nd-order DSM + ramp
  python dsm_ramp.py hspice ramp.csv [more.csv ...]       (several runs are averaged)
  python dsm_ramp.py dc steps.txt --step 1200 --skip 200   (stepped DC sweep)
  python dsm_ramp.py hspice ramp.csv [--vfsd 1.5]         HSPICE table: 4 columns time,v(qout),v(vip),v(vin) or 1 column v(qout) per clock

Requires numpy, scipy, matplotlib (numba optional, speeds up the modulator ~100x).
Full scale: +/-1 <-> Vcm +/- VFS (sim) or (vip-vin) = +/-VFSD (hspice); 16-bit signed output.
"""
import argparse
import numpy as np
from scipy import signal

try:
    from numba import njit
except ImportError:                       # slow pure-python fallback
    def njit(f):
        return f

# IDE users: set this to your data file (used when the script is run without arguments)
DATA_FILE = "af6859ramp1ms.txt"
NB = 16
LSB = 2.0 ** -(NB - 1)


@njit
def dsm2(u):
    """2nd-order 1-bit modulator (same structure as demo_dsm_test.m). u in FS, returns +/-1."""
    y = np.empty(u.size)
    i1 = 0.0
    i2 = 0.0
    v = 0.0
    for n in range(u.size):
        i1 += 0.5 * (u[n] - v)
        i2 += 0.5 * (i1 - v)
        v = 1.0 if i2 >= 0.0 else -1.0
        y[n] = v
    return y


def decimate(bits, fs, fc=10e3, r1=125, r2=2, cic_order=4, ntaps=161):
    """CIC(r1) + equiripple lowpass FIR (r2). Returns (y, fs_out, group_delay_s)."""
    box = np.ones(r1)
    cic = np.array([1.0])
    for _ in range(cic_order):
        cic = np.convolve(cic, box)
    cic /= cic.sum()
    fs1 = fs / r1
    lp = signal.remez(ntaps, [0, fc, 1.4 * fc, fs1 / 2], [1, 0], weight=[1, 5], fs=fs1)
    v = signal.oaconvolve(bits, cic)[:bits.size][::r1]
    y = signal.oaconvolve(v, lp)[:v.size][::r2]
    delay = cic_order * (r1 - 1) / 2 / fs + (ntaps - 1) / 2 / fs1
    return y, fs1 / r2, delay


def sinc3(bits, fs, r=128, order=3):
    """Short CIC for short HSPICE runs. Returns (y, t_index_of_samples, group_delay_s)."""
    h = np.array([1.0])
    for _ in range(order):
        h = np.convolve(h, np.ones(r))
    h /= h.sum()
    y = signal.oaconvolve(bits, h)[:bits.size]
    idx = np.arange(r - 1, bits.size, r)
    return y[idx], idx, order * (r - 1) / 2 / fs


def analyze(u_ideal, y, vfs=0.375, dnl=True, margin=50, fig_title="", vcm=0.75):
    """u_ideal, y in FS units at equal time instants. Prints results, returns dict."""
    code = np.clip(np.round(y / LSB), -2 ** (NB - 1), 2 ** (NB - 1) - 1)
    yq = code * LSB
    g, o = np.polyfit(u_ideal, yq, 1)
    err = yq - (g * u_ideal + o)
    res = dict(gain=g, offset=o, inl_pk=np.abs(err).max() / LSB, inl_rms=err.std() / LSB)
    print(f"Points: {u_ideal.size}")
    print(f"Gain   = {g:.5f}  ({100 * (g - 1):+.3f} % from 1)")
    print(f"Offset = {o:.3e} FS = {o / LSB:.1f} LSB")
    print(f"INL    = {res['inl_pk']:.2f} LSB peak, {res['inl_rms']:.2f} LSB rms (raw, incl. noise)")
    errs = err
    if err.size > 481:
        errs = np.convolve(err, np.ones(481) / 481, mode="same")
        c = slice(240, -240)
        print(f"INL (smoothed) = {np.abs(errs[c]).max() / LSB:.2f} LSB peak, "
              f"noise rms = {(err - errs)[c].std() / LSB:.2f} LSB")
    res["err"], res["errs"] = err, errs

    if dnl:
        ci = code.astype(int)
        lo, hi = ci.min() + margin, ci.max() - margin
        codes = np.arange(lo, hi + 1)
        cnt = np.bincount(ci[(ci >= lo) & (ci <= hi)] - lo, minlength=codes.size)
        hits = cnt.mean()
        d = cnt / hits - 1
        inl_h = np.cumsum(d)
        inl_h -= np.polyval(np.polyfit(codes, inl_h, 1), codes)
        nmiss = int((cnt == 0).sum())
        print(f"Codes {lo}..{hi} ({codes.size}), mean hits/code = {hits:.1f}")
        print(f"DNL: min {d.min():.2f}, max {d.max():.2f} LSB; INL(from DNL) pk {np.abs(inl_h).max():.2f} LSB")
        print(f"Missing codes: {nmiss} ({100 * nmiss / codes.size:.2f} %)")
        if hits < 10:
            print(f"WARNING: only {hits:.1f} hits/code, DNL/missing codes unreliable; increase ramp time.")
        res.update(codes=codes, dnl=d, nmiss=nmiss, hits=hits)

    import matplotlib
    import matplotlib.pyplot as plt
    n = 3 if dnl else 2
    mv = u_ideal * vfs * 1e3                      # input [mV]
    uv_lsb = vfs * LSB * 1e6                      # 1 LSB in uV (input referred)
    fig, ax = plt.subplots(n, 1, figsize=(8, 2.8 * n))
    ax[0].plot(mv, code, label="output")
    ax[0].plot(mv, (g * u_ideal + o) / LSB, "--", label="best fit")
    ax[0].set(xlabel="input (differential) [mV]", ylabel="output code [LSB, 16-bit]", title=fig_title)
    ax[0].legend()
    ax[1].plot(mv, err / LSB, ".-", ms=3, lw=0.8, label="INL (vs best-fit line)")
    if errs is not err:
        ax[1].plot(mv, errs / LSB, label="smoothed")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="input (differential) [mV]", ylabel="INL [LSB]",
              title=f"INL: {res['inl_pk']:.1f} LSB peak, {res['inl_rms']:.1f} LSB rms   "
                    f"(1 LSB = {uv_lsb:.1f} uV)")
    sec = ax[1].secondary_yaxis("right", functions=(lambda x: x * uv_lsb, lambda x: x / uv_lsb))
    sec.set_ylabel("INL [uV, input-referred]")
    ax[1].legend()
    if dnl:
        ax[2].plot(res["codes"], res["dnl"])
        ax[2].set(xlabel="output code [LSB]", ylabel="DNL [LSB]",
                  title=f"DNL (code density), {res['nmiss']} missing codes")
    for a_ in ax:
        a_.grid(True)
    fig.tight_layout()
    add_cursor(fig, ax[:2], mv, code, err / LSB, uv_lsb, vcm=vcm)
    return res


def add_cursor(fig, axes, mv, code, inl, uv_lsb, vcm=0.75):
    """Click in the top/middle plot (or press left/right arrows) to read out a point:
    input diff [mV] with vip/vin [V], output code, INL [LSB] and [uV].
    Both plots get a marker and a label (top: input/code, bottom: INL); each label can be
    dragged with the mouse."""
    import matplotlib

    def default_off(k, i):
        """Label offset [points] that stays inside its plot: top plot has empty space above-left
        and below-right of the rising line; the INL label goes on the side with more room."""
        right = mv[i] > mv.min() + 0.6 * np.ptp(mv)
        dx = -25 if right else 25
        if k == 0:
            return (dx, -55 if right else 35)
        return (dx, 30 if inl[i] < 0 else -45)

    st = {"i": 0, "art": [], "ann": [None, None], "off": [(0, 0), (0, 0)], "moved": [False, False]}

    def show(i):
        i = int(np.clip(i, 0, mv.size - 1))
        st["i"] = i
        for k, ann in enumerate(st["ann"]):
            if ann is not None and tuple(ann.xyann) != tuple(st["off"][k]):
                st["off"][k] = tuple(ann.xyann)           # remember where the user dragged it
                st["moved"][k] = True
        for art in st["art"]:
            art.remove()
        st["art"] = []
        vip, vin = vcm + mv[i] / 2e3, vcm - mv[i] / 2e3
        txts = (f"#{i}: diff {mv[i]:.2f} mV\n(vip {vip:.4f} V, vin {vin:.4f} V)\n"
                f"code {code[i]:+.0f} LSB",
                f"#{i}: diff {mv[i]:.2f} mV\nINL {inl[i]:+.2f} LSB = {inl[i] * uv_lsb:+.1f} uV")
        for k, (ax, yv) in enumerate(zip(axes, (code[i], inl[i]))):
            off = st["off"][k] if st["moved"][k] else default_off(k, i)
            right = off[0] < 0
            st["off"][k] = off
            ann = ax.annotate(
                txts[k], (mv[i], yv), xytext=off, textcoords="offset points",
                ha="right" if right else "left", fontsize=8, bbox=dict(boxstyle="round", fc="w", ec="r"),
                arrowprops=dict(arrowstyle="->", color="r"))
            ann.draggable(True)                           # drag each label with the mouse
            st["ann"][k] = ann
            st["art"] += [ax.axvline(mv[i], color="r", lw=0.8, ls=":"),
                          ax.plot(mv[i], yv, "ro", ms=5)[0], ann]
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.button != 1 or ev.x is None:
            return
        # Hit-test by pixel position instead of ev.inaxes: with a secondary (right-hand) axis,
        # some matplotlib versions report the overlay as inaxes, which broke clicks on the INL plot.
        if not any(ax.get_window_extent().contains(ev.x, ev.y) for ax in axes):
            return
        if any(a_ is not None and a_.contains(ev)[0] for a_ in st["ann"]):
            return                                        # clicking a label = drag, not select
        xdata = axes[0].transData.inverted().transform((ev.x, ev.y))[0]
        show(np.argmin(np.abs(mv - xdata)))

    def on_key(ev):
        if ev.key in ("left", "right"):
            show(st["i"] + (1 if ev.key == "right" else -1))

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._cursor_show = show            # for testing / scripting
    print("cursor: click in either plot or press left/right arrow keys (matplotlib %s)" % matplotlib.__version__)


def run_sim(a):
    fs = a.fs
    n = int(round(a.tr * fs))
    vin = np.linspace(a.vcm - a.vamp, a.vcm + a.vamp, n)
    u = (vin - a.vcm) / a.vfs
    bits = dsm2(u)
    y, fso, dly = decimate(bits, fs, a.fc)
    t = np.arange(y.size) / fso - dly
    ui = np.interp(t, np.arange(n) / fs, u, left=np.nan, right=np.nan)
    ok = ~np.isnan(ui) & (t > 5e-3) & (t < a.tr - 5e-3)
    return analyze(ui[ok], y[ok], dnl=a.tr >= 1, vfs=a.vfs,
                   fig_title=f"Behavioural 2nd-order DSM, fs={fs / 1e6:g} MHz")


def load_table(path):
    """Numeric rows of a CSV/WaveView text table (header lines skipped) and the
    signal names found in the header, e.g. ['v(vin)', 'v(vip)', 'v(qout)']."""
    rows, names = [], []
    with open(path) as f:
        for line in f:
            try:
                rows.append([float(v) for v in line.replace(",", " ").split()])
            except ValueError:
                names += [w.split(":")[0].lower() for w in line.replace(",", " ").split()
                          if w.lower().startswith("v(")]
    n = max(len(r) for r in rows)
    return np.array([r for r in rows if len(r) == n]), names


def run_hspice(a, path=None):
    d, names = load_table(path or a.csv[0])
    tck = 1.0 / a.fs
    ncol = d.shape[1]
    if ncol == 4 and not names:
        names = ["time", "v(qout)", "v(vip)", "v(vin)"]
    elif ncol == 1:
        names = ["v(qout)"]
    elif len(names) == ncol - 1:
        names = ["time"] + names                     # WaveView TIME column has no v(...) label
    col = {n: i for i, n in enumerate(names)}
    q = d[:, col["v(qout)"]]
    if "time" in col:
        t, i = np.unique(d[:, col["time"]], return_index=True)
        d, q = d[i], q[i]
        if abs(np.median(np.diff(t)) / tck - 1) < 0.01:
            # already one sample per clock: use the file's own sample instants
            ts = t
            bits = np.where(q > a.vth, 1.0, -1.0)
            vd = d[:, col["v(vip)"]] - d[:, col["v(vin)"]]
            if "v(clk_adcin)" in col:
                clk = d[:, col["v(clk_adcin)"]]
                ph = (ts % tck) / tck
                print(f"clk_adcin at samples: {np.mean(clk > 0.9):.1%} high; sample phase "
                      f"{ph.min():.3f}..{ph.max():.3f} of tck; qout in transition at samples: "
                      f"{np.mean((q > 0.2) & (q < 1.6)):.2%}")
        else:
            ts = np.arange(a.tsamp, t[-1], tck)
            ts = ts[ts >= t[0]]
            bits = np.where(np.interp(ts, t, q) > a.vth, 1.0, -1.0)
            vd = np.interp(ts, t, d[:, col["v(vip)"]] - d[:, col["v(vin)"]])
    else:
        # one row per clock (.tran 'tck' ... start='tck'): no time column
        bits = np.where(q > a.vth, 1.0, -1.0)
        ts = (np.arange(bits.size) + 1) * tck
        if "v(vip)" in col and "v(vin)" in col:
            vd = d[:, col["v(vip)"]] - d[:, col["v(vin)"]]
        else:
            vd = a.vd0 + (a.vd1 - a.vd0) * ts / a.tramp      # known netlist ramp
    u = vd / a.vfsd
    print(f"columns: {names}; diff input {vd[0]:.4f} -> {vd[-1]:.4f} V")
    y, idx, dly = sinc3(bits, a.fs, a.r)
    ui = np.interp(ts[idx] - dly, ts, u)
    keep = np.arange(y.size) > 4
    print(f"{bits.size} bits, R={a.r}, fout={a.fs / a.r / 1e3:.1f} kHz, VfsD={a.vfsd} V assumed "
          f"(wrong VfsD only scales gain), mean bit = {bits.mean():+.4f}. Too few points for DNL.")
    return ui[keep], y[keep]


def run_hspice_files(a):
    """One or more ramp runs (e.g. different noise seeds): decimated outputs are averaged."""
    runs = [run_hspice(a, f) for f in a.csv]
    u = runs[0][0]
    y = np.mean([r[1] for r in runs], axis=0)
    if len(runs) > 1:
        print(f"averaged {len(runs)} runs")
    return analyze(u, y, vfs=a.vfsd, dnl=False, fig_title=f"HSPICE ramp (sinc3), {len(runs)} run(s)")


def run_dc(a):
    """Stepped-input (DC sweep) run: each step lasts --step clocks; the first --skip are
    discarded, the rest averaged. Columns: [TIME] v(vin) v(vip) v(qout), one row per clock."""
    d, names = load_table(a.csv[0])
    if len(names) == d.shape[1] - 1:
        names = ["time"] + names
    col = {n: i for i, n in enumerate(names)}
    q = d[:, col["v(qout)"]]
    bits = np.where(q > a.vth, 1.0, -1.0)
    vd = d[:, col["v(vip)"]] - d[:, col["v(vin)"]]
    nst = bits.size // a.step
    # sinc^3 weighting over the usable part of each step (a plain average of a 2nd-order
    # DSM stream leaves ~1/M error, which would show up as INL)
    L = (a.step - a.skip - 1) // 3 + 1
    h = np.array([1.0])
    for _ in range(3):
        h = np.convolve(h, np.ones(L))
    h /= h.sum()
    us, ys = [], []
    for k in range(nst):
        sl = slice((k + 1) * a.step - h.size, (k + 1) * a.step)
        us.append(vd[sl].mean() / a.vfsd)
        ys.append(bits[sl] @ h)
    print(f"{nst} steps x {a.step} clocks ({a.skip} skipped), diff {min(vd):.4f}..{max(vd):.4f} V, "
          f"VfsD={a.vfsd} V assumed")
    return analyze(np.array(us), np.array(ys), vfs=a.vfsd, dnl=False, fig_title=f"DC sweep, {nst} steps")




def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    s = p.add_subparsers(dest="cmd")
    a1 = s.add_parser("sim")
    a1.add_argument("--fs", type=float, default=12e6)
    a1.add_argument("--fc", type=float, default=10e3)
    a1.add_argument("--tr", type=float, default=4.0, help="ramp time [s]; >=1 enables DNL")
    a1.add_argument("--vcm", type=float, default=0.75)
    a1.add_argument("--vamp", type=float, default=0.15)
    a1.add_argument("--vfs", type=float, default=0.75)
    a2 = s.add_parser("hspice", help="ramp run(s); several files are averaged")
    a2.add_argument("csv", nargs="+")
    a3 = s.add_parser("dc", help="stepped-input (DC sweep) run, see ramp_steps.sp")
    a3.add_argument("csv", nargs=1)
    a3.add_argument("--step", type=int, default=1200, help="clocks per input step")
    a3.add_argument("--skip", type=int, default=200, help="clocks discarded at start of each step")
    a3.add_argument("--vfsd", type=float, default=0.375)
    a3.add_argument("--vth", type=float, default=0.9)
    a2.add_argument("--fs", type=float, default=12e6)
    a2.add_argument("--tsamp", type=float, default=100e-12 + 0.75 / 12e6)
    a2.add_argument("--vth", type=float, default=0.9)
    a2.add_argument("--vfsd", type=float, default=0.375, help="diff full scale [V] = Vref = VLDOA/16*4")
    a2.add_argument("--r", type=int, default=128)
    a2.add_argument("--vd0", type=float, default=0.0, help="1-column input: diff at t=0 [V]")
    a2.add_argument("--vd1", type=float, default=0.3, help="1-column input: diff at t=tramp [V]")
    a2.add_argument("--tramp", type=float, default=1e-3)
    for sp in (a1, a2, a3):
        sp.add_argument("--save", help="save the figure to this file instead of showing it")
    import os
    import sys
    if not {"sim", "hspice", "dc"} & set(sys.argv[1:]):
        # No subcommand (e.g. run from an IDE, whose argv may carry extra junk): use defaults.
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = os.getcwd()
        cands = [DATA_FILE, os.path.join(os.getcwd(), DATA_FILE), os.path.join(here, DATA_FILE)]
        f = next((c for c in cands if os.path.exists(c)), None)
        sys.argv = sys.argv[:1] + (["hspice", f] if f else ["sim", "--tr", "0.5"])
    a = p.parse_args()
    import matplotlib.pyplot as plt
    {"sim": run_sim, "hspice": run_hspice_files, "dc": run_dc}[a.cmd](a)
    out = a.save or (os.path.splitext(a.csv[0])[0] + "_inl.png" if a.cmd != "sim" else "dsm_sim_inl.png")
    plt.gcf().savefig(out, dpi=130)
    print(f"figure saved: {os.path.abspath(out)}")
    if not a.save:
        plt.show()


if __name__ == "__main__":
    main()
