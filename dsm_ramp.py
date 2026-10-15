#!/usr/bin/env python3
"""Ramp test of a 1-bit delta-sigma ADC (gain / offset / INL / DNL / missing codes).

  python dsm_ramp.py sim    [--tr 4] [--fs 12e6]          behavioural 2nd-order DSM + ramp
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


def analyze(u_ideal, y, vfs_label="FS", dnl=True, margin=50, fig_title=""):
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

    import matplotlib.pyplot as plt
    n = 4 if dnl else 3
    fig, ax = plt.subplots(n, 1, figsize=(8, 2.6 * n))
    ax[0].plot(u_ideal, yq, label="output")
    ax[0].plot(u_ideal, g * u_ideal + o, "--", label="best fit")
    ax[0].set(xlabel="input [FS]", ylabel="output [FS]", title=fig_title)
    ax[0].legend()
    ax[1].plot(u_ideal, err / LSB, lw=0.6, label="raw")
    ax[1].plot(u_ideal, errs / LSB, label="smoothed")
    ax[1].set(xlabel="input [FS]", ylabel="INL [LSB]")
    ax[1].legend()
    if dnl:
        ax[2].plot(res["codes"], res["dnl"])
        ax[2].set(xlabel="output code", ylabel="DNL [LSB]",
                  title=f"DNL (code density), {res['nmiss']} missing codes")
    for a in ax:
        a.grid(True)
    fig.tight_layout()
    return res


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
    return analyze(ui[ok], y[ok], dnl=a.tr >= 1,
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


def run_hspice(a):
    d, names = load_table(a.csv)
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
    return analyze(ui[keep], y[keep], dnl=False, fig_title="HSPICE ramp (sinc3)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    a1 = s.add_parser("sim")
    a1.add_argument("--fs", type=float, default=12e6)
    a1.add_argument("--fc", type=float, default=10e3)
    a1.add_argument("--tr", type=float, default=4.0, help="ramp time [s]; >=1 enables DNL")
    a1.add_argument("--vcm", type=float, default=0.75)
    a1.add_argument("--vamp", type=float, default=0.15)
    a1.add_argument("--vfs", type=float, default=0.75)
    a2 = s.add_parser("hspice")
    a2.add_argument("csv")
    a2.add_argument("--fs", type=float, default=12e6)
    a2.add_argument("--tsamp", type=float, default=100e-12 + 0.75 / 12e6)
    a2.add_argument("--vth", type=float, default=0.9)
    a2.add_argument("--vfsd", type=float, default=0.375, help="diff full scale [V] = Vref = VLDOA/16*4")
    a2.add_argument("--r", type=int, default=128)
    a2.add_argument("--vd0", type=float, default=0.0, help="1-column input: diff at t=0 [V]")
    a2.add_argument("--vd1", type=float, default=0.3, help="1-column input: diff at t=tramp [V]")
    a2.add_argument("--tramp", type=float, default=1e-3)
    a = p.parse_args()
    import matplotlib.pyplot as plt
    (run_sim if a.cmd == "sim" else run_hspice)(a)
    plt.show()


if __name__ == "__main__":
    main()
