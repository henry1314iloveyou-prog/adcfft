"""CIC + lowpass FIR decimation of a 1-bit stream (port of dsm_decimate.m)."""
import numpy as np
from scipy import signal


def decimate(bits, fs, fc, r1=125, r2=2, cic_order=4, stop_edge=None, ntaps=161):
    """Return (y, fs_out, coeffs). Defaults: 12 MHz -> /125 -> 96 kHz -> /2 -> 48 kHz."""
    stop_edge = 1.4 * fc if stop_edge is None else stop_edge
    x = np.asarray(bits, dtype=float).ravel()
    if np.all((x == 0) | (x == 1)):
        x = 2 * x - 1

    cic = np.ones(1)
    for _ in range(cic_order):
        cic = np.convolve(cic, np.ones(r1))
    cic /= cic.sum()
    fs1 = fs / r1
    lp = signal.remez(ntaps, [0, fc, stop_edge, fs1 / 2], [1, 0], weight=[1, 5], fs=fs1)

    v = signal.lfilter(cic, 1, x)[::r1]
    y = signal.lfilter(lp, 1, v)[::r2]
    return y, fs1 / r2, {"cic": cic, "lp": lp}
