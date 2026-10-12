"""Overlay the PWL Gaussian-noise injection runs (70/80/90 dB) and the noiseless baseline (N=16384, M=5).
GUI friendly: edit the settings and press Run (F5).
Keep this file, dsm_fft.py and dsm_compare.py together with the data files."""
import os
import sys

# ---------------- settings ----------------
# (file, label, injected noise-only SNR in dB or None).  The reference run is the one at REF.
FILES = [
    ("adcfft_70dbpwlnoise.txt", "injected 70 dB",      70),   # A
    ("adcfft_80dbpwlnoise.txt", "injected 80 dB",      80),   # B
    ("adcfft_90dbpwlnoise.txt", "injected 90 dB",      90),   # C
    ("adcfft_1151002a.txt",      "baseline (no noise)", None), # D
]
REF = -1                   # index of the reference run (-1 = the last one = baseline)
FS      = 12e6
OSR     = 256
SKIP    = 200
OUT     = "fft_compare_injected_noise_16384.png"
N       = 16384            # same N / FIN for ALL files:  N=16384 -> M=5, FIN=3662.109375
FIN     = 3662.109375      #                              N=32768 -> M=7, FIN=2563.4765625
WINDOW  = "bh"             # bh | hann
BW      = None             # None = FS/(2*OSR) = 23437.5;  e.g. 24300
THRESH  = None             # None = auto (two-rail voltage -> 0.9 V)
# ------------------------------------------

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)
import dsm_compare

args = [f for f, _, _ in FILES] + [str(FS), str(OSR),
        "--labels", ",".join(l for _, l, _ in FILES), "--ref", str(REF), "--out", OUT]
if any(i is not None for _, _, i in FILES):
    args += ["--inject", ",".join("-" if i is None else str(i) for _, _, i in FILES)]
opts = ["--skip", str(SKIP), "--n", str(N), "--fin", str(FIN), "--window", WINDOW]
if BW is not None:
    opts += ["--bw", str(BW)]
if THRESH is not None:
    opts += ["--thresh", str(THRESH)]
dsm_compare.main(args + opts)
