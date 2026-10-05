"""FFT analysis of a 1-bit delta-sigma ADC stream: raw, and optionally after CIC and/or FIR decimation.

Usage: python dsm_fft.py [capture.txt] [fs] [OSR] [cic_order] [--skip S] [--n N] [--fin F]
                                  [--window bh|hann|rect]
                                  [--cic R] [--cic-order N] [--fir R2] [--fir-fc F] [--fir-stop F] [--fir-atten dB]
                                  [--thresh V] [--resample 1] [--phase P] [--col C] [--ignore-time 1] [--cursor 0] [--bw Hz] [--nharm N] [--noplot 1]
  --skip S  drop the first S samples (start-up transient), default 0
  --n N     use N contiguous samples after the skip (default: all, cut to a
            multiple of OSR so the CIC output is whole)
  --fin F   expected input frequency [Hz]; checks coherence (M=fin*N/fs integer,
            gcd(M,N)=1). Coherent example: fs=12 MHz, N=65536, M=11 -> 2014.16015625 Hz
  --window  bh (Blackman-Harris, default, signal = +/-4 bins), hann (+/-2 bins),
            rect (signal = 1 bin; only for coherent sampling, noise-shaped
            high-freq noise may leak in-band)
  --cic R       enable the CIC stage with decimation ratio R (0 = off, default).
                `--cic-order N` sets the order (default = 4th positional arg, 3).
  --fir R2      enable the FIR low-pass stage, decimating by R2 (0 = off, default).
                It follows the CIC if that is on, otherwise it filters the 1-bit stream.
                --fir-fc F     passband edge [Hz]; default 0.4*fs_out (R2>1) or the signal
                               bandwidth fs/(2*OSR). SNR/ENOB of the decimated spectrum are
                               then computed over 0..fc (only the passband is meaningful).
                --fir-stop F   stopband edge [Hz], default = fs_out - fc (nothing aliases
                               into the passband); with R2=1 default 1.4*fc
                --fir-atten dB stopband attenuation of the Kaiser design, default 70
                Typical: --cic 128 --fir 2   (12 MHz -> 93.75 kHz -> 46.875 kHz)
                --dec-window W window for the decimated spectrum (bh default: the short,
                               non-coherent decimated record leaks badly with hann)
                Both off -> only the raw 1-bit spectrum. (--nocic 0 = old behaviour:
                CIC with R = OSR; --nocic 1 = off.)
  --thresh V    input is an analog voltage column; threshold it at V (e.g. 0.9) to 0/1
  --resample 1  two-column (time, value) input: re-sample at t0 + k/fs by linear
                interpolation before thresholding (use when the time step is not 1/fs)
  --phase P     with --resample: sample at P + k/fs [s] (e.g. 62.5e-9 = 10p+0.75*tck);
                default = the first time point in the file
  --ignore-time 1  two-column file whose time column is too coarse (few digits) but whose
                rows are exactly one sample per clock: use the values in order
  --col C       if the file has more than 2 columns: which column (0-based) is v(qout), default 1
  --bw Hz       signal bandwidth for SNR/ENOB/SFDR; default fs/(2*OSR). Use it when the
                bandwidth is not exactly fs/(2*OSR), e.g. --bw 24300
  --nharm N     number of harmonics (HD2..HD(N+1)) counted in THD / excluded from SNR, default 8
  --noplot 1    analyse only (no figure); used by dsm_compare.py
  --cursor 0    disable the interactive cursor line (default on when a plot window opens):
                move the mouse = vertical cursor line + f / dBFS readout (snaps to the
                data); left click = drop marker M1, M2 (shows delta f / delta dB);
                right click or 'c' = clear markers; left/right arrow keys = step the
                cursor one bin; 'm' = drop a marker at the cursor.
                Not active while the toolbar zoom/pan tool is selected.
Needs: numpy, matplotlib   (pip install numpy matplotlib)

The CIC: `order` integrators at fs, decimate by R, `order` combs at fs/R, unity DC gain
(deci_hspice.mdl uses R = OSR = 256). The FIR is a Kaiser-window low-pass (numpy only);
only fully-settled output samples are kept, so short records lose ~taps/R2 points.
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import collections
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
nharm = int(opts.get("nharm", 8))
if window not in WINDOWS:
    sys.exit("--window must be bh, hann or rect")
if "cic" in opts:
    cic_R = int(float(opts["cic"]))
elif opts.get("nocic") in ("0", "false"):
    cic_R = osr                           # old behaviour
else:
    cic_R = 0
order = int(opts.get("cic-order", order))
fir_R = int(float(opts.get("fir", 0)))
bw = float(opts["bw"]) if "bw" in opts else fs / (2 * osr)   # signal bandwidth [Hz]
print(f"signal bandwidth = {bw:g} Hz" + ("" if "bw" in opts else f"  (fs/(2*OSR), OSR={osr})"))


def cic_decimate(x, R, order):
    """CIC decimator (integrators -> downsample by R -> combs), unity DC gain."""
    y = x.astype(np.float64)
    for _ in range(order):
        y = np.cumsum(y)                  # integrators at fs
    y = y[R - 1::R]                       # decimate
    for _ in range(order):
        y = np.diff(y, prepend=0.0)       # combs at fs/R
    return y / R ** order


def fir_lowpass(fs_in, fc, fstop, atten_db):
    """Kaiser-window low-pass: passband edge fc, stopband edge fstop, unity DC gain."""
    A = atten_db
    beta = (0.1102 * (A - 8.7) if A > 50 else
            0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21) if A > 21 else 0.0)
    ntaps = int(np.ceil((A - 7.95) / (2.285 * 2 * np.pi * (fstop - fc) / fs_in))) + 1
    ntaps += 1 - ntaps % 2                # odd length
    fcut = (fc + fstop) / 2 / fs_in
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = 2 * fcut * np.sinc(2 * fcut * n) * np.kaiser(ntaps, beta)
    return h / h.sum()


def make_window(kind, N):
    n = np.arange(N)
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)
    if kind == "rect":
        return np.ones(N)
    return (0.35875 - 0.48829 * np.cos(2 * np.pi * n / N)
            + 0.14128 * np.cos(4 * np.pi * n / N) - 0.01168 * np.cos(6 * np.pi * n / N))


def analyze(v, fs, bw, name, fin_expect=None, win=None):
    """Return dict of spectrum + metrics. 0 dB = full-scale sine (amplitude 1)."""
    wn = win or window
    N = len(v)
    w = make_window(wn, N)
    v = v - np.sum(w * v) / np.sum(w)     # remove DC as seen through the window
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

    hw = WINDOWS[wn]
    k0 = int(round(fin / df))
    sig = np.zeros(len(P), bool)
    sig[max(k0 - hw, 0):k0 + hw + 1] = True
    mask = np.zeros(len(P), bool)
    mask[WINDOWS[wn] + 2:bwb + 1] = True   # skip the DC lobe of the window
    mask &= ~sig
    Ps = P[sig].sum()
    enbw = N * np.sum(w ** 2) / np.sum(w) ** 2     # window noise bandwidth [bins]

    # harmonics of the tone, folded into 0..fs/2; only those inside the band and not
    # overlapping the signal lobe are counted
    hmask = np.zeros(len(P), bool)
    harm = []
    for h in range(2, nharm + 2):
        fh = (h * fin) % fs
        fh = fs - fh if fh > fs / 2 else fh
        kh = int(round(fh / df))
        if hw + 2 <= kh - hw and kh + hw <= bwb and kh - hw > k0 + hw:
            hmask[kh - hw:kh + hw + 1] = True
            harm.append((h, fh, 10 * np.log10(4 * P[max(kh - 1, 0):kh + 2].max() + 1e-30)))
    hmask &= mask
    Pn_all = P[mask].sum()                            # noise + distortion
    Ph = P[hmask].sum()                               # harmonic lobes (noise inside included)
    peaks = np.zeros(len(P), bool)                    # harmonic centres +-1 bin
    for h, fh, _ in harm:
        kh = int(round(fh / df))
        peaks[max(kh - 1, 0):kh + 2] = True
    clean = mask & ~peaks
    floor = np.median(P[clean]) if clean.any() else np.median(P[mask])   # robust noise floor/bin
    # noise only: remove each harmonic lobe, but put back the noise floor that sat in those bins
    Pn_nh = max(Pn_all - Ph + floor * hmask.sum(), 1e-30)
    sinad = 10 * np.log10(Ps / Pn_all)
    snr = sinad                                       # kept for the plots / older code
    snr_nh = 10 * np.log10(Ps / Pn_nh)
    thd = 10 * np.log10(max(Ph - floor * hmask.sum(), 1e-30) / Ps) if harm else float("nan")
    nd = 10 * np.log10(4 * floor / (enbw * df))       # noise density, dBFS/Hz (+-3 dB, see note)
    n_floor = int(clean.sum())
    r = dict(name=name, win=wn, bw=bw, N=N, fs=fs, f=f, d=10 * np.log10(4 * P + 1e-30), fin=fin,
             amp=10 * np.log10(4 * Ps / enbw), snr=snr, sinad=sinad, snr_nh=snr_nh, thd=thd,
             harm=harm, nd=nd, n_floor=n_floor, enob=(sinad - 1.76) / 6.02,
             sfdr=10 * np.log10(Ps / P[mask].max()), df=df)
    print(f"[{name}] N={N} fs={fs:g} Hz bin={df:.1f} Hz | fin={fin:.1f} Hz "
          f"({r['amp']:.2f} dBFS) SINAD={sinad:.2f} dB  SNR(excl. harmonics)={snr_nh:.2f} dB  "
          f"THD={thd:.1f} dB  ENOB={r['enob']:.2f} bit  SFDR={r['sfdr']:.1f} dB")
    if harm:
        print(f"[{name}]   harmonics: " + "  ".join(f"HD{h} {fh / 1e3:.2f}k {lv:.1f}dBFS"
                                                   for h, fh, lv in harm))
    print(f"[{name}]   in-band noise floor (median of {n_floor} bins, harmonic peaks excluded): "
          f"{nd:.1f} dBFS/Hz  -- only +-3 dB accurate for records this short; do not use it "
          f"to compare runs")
    return r


def load_table(path):
    """Read numbers from a text/CSV file: any of space, tab, comma, semicolon as
    separators; lines that are not all-numeric (headers, units) are skipped."""
    rows, skipped = [], 0
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                skipped += 1
                continue
    rows = [r for r in rows if r]
    if not rows:
        sys.exit("ERROR: no numeric data found in " + path)
    width = collections.Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == width]
    if skipped:
        print(f"skipped {skipped} non-numeric line(s) (header/units)")
    a = np.array(rows)
    return a[:, 0] if width == 1 else a


x = load_table(path)                      # one 0/1 per line, or (time, value[, ...]) columns
if x.ndim == 2 and x.shape[1] > 2:
    col = int(opts.get("col", 1))
    print(f"file has {x.shape[1]} columns; using column {col} as the signal "
          f"(change with --col)")
    x = x[:, [0, col]]
if x.ndim == 2 and opts.get("ignore-time", "0") not in ("0", "false"):
    # rows are one sample per clock (e.g. HSPICE `.option interp` + tstep = tck):
    # trust the ORDER, not the (possibly low-precision) time column
    t_col = x[:, 0]
    x = x[:, 1]
    print(f"ignoring the time column: {len(x)} rows taken as one sample per clock "
          f"(time column spans {t_col[0]:.4g} .. {t_col[-1]:.4g} s, "
          f"expected ~{(len(x) - 1) / fs:.4g} s)")
    if abs((t_col[-1] - t_col[0]) * fs / (len(x) - 1) - 1) > 0.01:
        sys.exit("ERROR: the row count does not match the time span at one row per clock; "
                 "the file is not one-sample-per-clock -- do not use --ignore-time")
if x.ndim == 2:                           # two columns: time, value
    t, x = x[:, 0], x[:, 1]
    if "resample" in opts and opts["resample"] not in ("0", "false"):
        p0 = float(opts["phase"]) if "phase" in opts else t[0]
        k0 = int(np.ceil((t[0] - p0) * fs - 1e-9)) if p0 < t[0] else 0
        tn = p0 + (k0 + np.arange(int((t[-1] - p0) * fs) + 1 - k0)) / fs
        print(f"sampling instants: first {tn[0]:.4e} s (phase {tn[0] * fs % 1 / fs * 1e9:.2f} ns "
              f"after a multiple of 1/fs)")
        x = np.interp(tn, t, x)
        t = tn
        print(f"resampled to {len(x)} points at 1/fs")
    dt = np.diff(t)
    print(f"time column: median step {np.median(dt):.4g} s "
          f"(expected {1 / fs:.4g} s), min {dt.min():.4g}, max {dt.max():.4g}")
    if abs(np.median(dt) * fs - 1) > 1e-3 or dt.max() > 1.5 / fs:
        sys.exit("ERROR: time step is not 1/fs -- the file is not one value per clock "
                 "(WaveView may have exported only the transitions); resample first")
if "thresh" in opts:
    x = (x > float(opts["thresh"])).astype(float)
elif len(np.unique(x)) > 2:
    # a voltage waveform: accept it only if it is clearly two-level (rails), then
    # threshold at the mid-point; anything else is probably the wrong file/column
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    span = hi - lo
    near = np.mean((np.abs(x - lo) < 0.1 * span) | (np.abs(x - hi) < 0.1 * span))
    if span >= 0.5 and near >= 0.95:
        mid = round((lo + hi) / 2, 3)
        print(f"voltage waveform with two rails ({lo:.3g} V / {hi:.3g} V): "
              f"thresholding at {mid} V automatically (set THRESH / --thresh to override)")
        x = (x > mid).astype(float)
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
fc = bw
if cic_R > 0 or fir_R > 0:
    y, fs_o, label = v, fs, []
    if cic_R > 0:
        y = cic_decimate(y, cic_R, order)[order + 2:]    # drop CIC start-up transient
        fs_o = fs / cic_R
        label.append(f"CIC{order} /{cic_R}")
        print(f"CIC: order {order}, R = {cic_R}, fs_out = {fs_o:g} Hz, {len(y)} samples")
    if fir_R > 0:
        fs_f = fs_o / fir_R
        fc = float(opts["fir-fc"]) if "fir-fc" in opts else (0.4 * fs_f if fir_R > 1 else bw)
        fstop = float(opts["fir-stop"]) if "fir-stop" in opts else (
            fs_f - fc if fir_R > 1 else min(1.4 * fc, 0.49 * fs_o))
        if not fc < fstop < fs_o / 2:
            sys.exit(f"ERROR: need fc ({fc:g}) < stopband edge ({fstop:g}) < {fs_o / 2:g} Hz; "
                     f"adjust --fir-fc / --fir-stop / --fir")
        h = fir_lowpass(fs_o, fc, fstop, float(opts.get("fir-atten", 70)))
        if len(y) < len(h):
            sys.exit(f"ERROR: FIR has {len(h)} taps but only {len(y)} samples reach it; "
                     f"use a longer record, a smaller --cic, or lower --fir-atten")
        y = np.convolve(y, h, "valid")[::fir_R]          # settled samples only
        fs_o = fs_f
        label.append(f"FIR /{fir_R} (fc={fc / 1e3:g}k)")
        print(f"FIR: {len(h)} taps, passband {fc:g} Hz, stopband {fstop:g} Hz, "
              f"fs_out = {fs_o:g} Hz, {len(y)} samples")
    if len(y) < 32:
        print(f"warning: only {len(y)} output samples -- the decimated spectrum is very "
              f"coarse; use a longer record or analyse the raw stream")
    bw_dec = min(bw, fc) if fir_R > 0 else bw
    if bw_dec < bw:
        print(f"decimated spectrum: SNR/ENOB computed over 0..{bw_dec:g} Hz (FIR passband)")
    dec = analyze(y, fs_o, bw_dec, " + ".join(label), fin_exp, opts.get("dec-window", "bh"))

class CursorTool:
    """Vertical cursor line with f/dBFS readout, two markers and delta readout."""

    def __init__(self, fig, panels):
        # panels: {axes: [(label, x, y), ...]}; the first entry is the reference trace
        self.fig, self.panels = fig, panels
        self.cur_ax, self.cur_x = None, None
        self.marks = []                               # [(ax, x, y, artists)]
        self.vl = {a: a.axvline(1, c="0.35", lw=0.8, visible=False) for a in panels}
        self.note = {a: a.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                                   fontsize=8, visible=False,
                                   bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
                     for a in panels}
        self.info = fig.text(0.01, 0.003, "", fontsize=8, family="monospace", va="bottom")
        self.help = fig.text(0.99, 0.003, "mouse: cursor | click: marker | right-click/c: clear"
                             " | arrows: step | m: marker", fontsize=7, ha="right",
                             va="bottom", color="0.4")
        for ev, fn in (("motion_notify_event", self.move), ("button_press_event", self.click),
                       ("key_press_event", self.key)):
            fig.canvas.mpl_connect(ev, fn)

    @staticmethod
    def _nearest(ax, x, xs):
        if ax.get_xscale() == "log":
            return int(np.argmin(np.abs(np.log10(np.maximum(xs, 1e-30)) - np.log10(max(x, 1e-30)))))
        return int(np.argmin(np.abs(xs - x)))

    def _busy(self):                                  # zoom / pan tool active?
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(tb is not None and str(getattr(tb, "mode", "")))

    def _show(self, ax, x):
        traces = self.panels[ax]
        _, xr, yr = traces[0]
        i = self._nearest(ax, x, xr)
        self.cur_ax, self.cur_x, self.cur_i = ax, xr[i], i
        lines = []
        for lab, xs, ys in traces:
            j = self._nearest(ax, xr[i], xs)
            lines.append(f"{lab + ': ' if lab else ''}f={xs[j]:.1f} {self.unit(ax)}  {ys[j]:.1f} dBFS")
        self.vl[ax].set_xdata([xr[i], xr[i]])
        for a in self.panels:
            self.vl[a].set_visible(a is ax)
            self.note[a].set_visible(a is ax)
        self.note[ax].xy = (xr[i], yr[i])
        self.note[ax].set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    @staticmethod
    def unit(ax):
        return "kHz" if ax.get_xlabel().lower().startswith("khz") else "Hz"

    def move(self, ev):
        if ev.inaxes in self.panels and ev.xdata is not None and not self._busy():
            self._show(ev.inaxes, ev.xdata)

    def key(self, ev):
        if ev.key in ("left", "right") and self.cur_ax is not None:
            xs = self.panels[self.cur_ax][0][1]
            i = min(max(self.cur_i + (1 if ev.key == "right" else -1), 0), len(xs) - 1)
            self._show(self.cur_ax, xs[i])
        elif ev.key == "m" and self.cur_ax is not None:
            self.mark()
        elif ev.key == "c":
            self.clear()

    def click(self, ev):
        if self._busy() or ev.inaxes not in self.panels:
            return
        if ev.button == 3:
            self.clear()
        elif ev.button == 1:
            self._show(ev.inaxes, ev.xdata)
            self.mark()

    def mark(self):
        ax = self.cur_ax
        _, xs, ys = self.panels[ax][0]
        if len(self.marks) >= 2:
            self.clear()
        art = ax.plot([xs[self.cur_i]], [ys[self.cur_i]], "v", c="C3", ms=7)[0]
        txt = ax.annotate(f"M{len(self.marks) + 1}", (xs[self.cur_i], ys[self.cur_i]),
                          xytext=(0, 8), textcoords="offset points", ha="center",
                          fontsize=8, color="C3")
        self.marks.append((ax, xs[self.cur_i], ys[self.cur_i], (art, txt)))
        self._readout()

    def clear(self):
        for *_, arts in self.marks:
            for a in arts:
                a.remove()
        self.marks = []
        self._readout()

    def _readout(self):
        parts = [f"M{k + 1}: {x:.1f} {self.unit(a)}, {y:.1f} dBFS"
                 for k, (a, x, y, _) in enumerate(self.marks)]
        if len(self.marks) == 2:
            (a1, x1, y1, _), (a2, x2, y2, _) = self.marks
            if a1 is a2:
                parts.append(f"delta f = {x2 - x1:.1f} {self.unit(a1)}, delta = {y2 - y1:.1f} dB")
        self.info.set_text("   ".join(parts))
        self.fig.canvas.draw_idle()


NOPLOT = opts.get("noplot", "0") not in ("0", "false")
if not NOPLOT:
    rs = [raw] + ([dec] if dec else [])
    fig, axs = plt.subplots(len(rs) + 1, 2, figsize=(14, 4.3 * (len(rs) + 1)),
                            gridspec_kw={"width_ratios": [3.3, 1]})
    ax, side = axs[:, 0], axs[:, 1]               # spectra on the left, numbers on the right
    for a in side:
        a.axis("off")
    def metrics_text(r):
        """Multi-line summary of one analysis for the plot."""
        lines = [f"{r['name']}   [{r['win']} window]",
                 f"fs = {r['fs'] / 1e3:g} kHz   N = {r['N']}   bin = {r['df']:.1f} Hz",
                 f"BW = {r['bw'] / 1e3:.2f} kHz",
                 f"fin  = {r['fin']:.1f} Hz   {r['amp']:.2f} dBFS",
                 f"SINAD = {r['sinad']:.2f} dB",
                 f"SNR   = {r['snr_nh']:.2f} dB  (excl. harmonics)",
                 f"THD   = {r['thd']:.1f} dB",
                 f"ENOB  = {r['enob']:.2f} bit",
                 f"SFDR  = {r['sfdr']:.1f} dB",
                 f"noise floor = {r['nd']:.1f} dBFS/Hz (+-3 dB)"]
        if r["harm"]:
            lines.append("harmonics [dBFS]: " + " ".join(f"HD{h}={lv:.0f}" for h, _, lv in r["harm"][:6]))
        return "\n".join(lines)


    panels = {}
    for r, a in zip(rs, ax):
        a.semilogx(r["f"][1:], r["d"][1:], lw=0.7)
        panels[a] = [("", r["f"][1:], r["d"][1:])]
        a.axvline(r["bw"], c="r", ls="--")
        a.set(xlabel="Frequency [Hz]", ylabel="dBFS", ylim=(-140, 0), title=r["name"])
        a.grid(True, which="both", alpha=0.3)
    for r, sa in zip(rs, side):
        sa.text(0.0, 1.0, metrics_text(r), transform=sa.transAxes, ha="left", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", fc="0.97", ec="0.6"))
    for r, lab in zip(rs, ("raw", dec["name"] if dec else "")):
        m = r["f"] <= bw * 1.2
        ax[-1].plot(r["f"][m] / 1e3, r["d"][m], "o-", ms=3, lw=0.8, label=lab)
        panels.setdefault(ax[-1], []).append((lab, r["f"][m] / 1e3, r["d"][m]))
    ax[-1].axvline(bw / 1e3, c="r", ls="--")
    ax[-1].set(xlabel="kHz", ylabel="dBFS", ylim=(-140, 0), title="In-band")
    ax[-1].legend(loc="lower left", fontsize=8)
    ax[-1].grid(alpha=0.3)
    # comparison table of all analyses (right of the in-band panel)
    cols = ["raw"] + (["decimated"] if dec else [])
    rows = [("SINAD [dB]", "sinad", "{:.2f}"), ("SNR excl. harm. [dB]", "snr_nh", "{:.2f}"),
            ("THD [dB]", "thd", "{:.1f}"), ("ENOB [bit]", "enob", "{:.2f}"),
            ("SFDR [dB]", "sfdr", "{:.1f}"), ("fin [Hz]", "fin", "{:.1f}"),
            ("amplitude [dBFS]", "amp", "{:.2f}"), ("noise floor [dBFS/Hz]", "nd", "{:.1f}")]
    tab = side[-1].table(cellText=[[fmt.format(r[key]) for r in rs] for _, key, fmt in rows],
                         rowLabels=[lab for lab, _, _ in rows], colLabels=cols,
                         cellLoc="center", loc="center", bbox=[0.55, 0.15, 0.45, 0.7])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig("fft_result.png", dpi=110)
    if opts.get("cursor", "1") not in ("0", "false") and (
            matplotlib.get_backend().lower() != "agg" or opts.get("cursor") == "force"):
        cursor = CursorTool(fig, panels)              # keep a reference alive
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
