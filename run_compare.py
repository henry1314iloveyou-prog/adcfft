"""Overlay two runs (e.g. with / without transient noise) in one FFT plot -- GUI friendly.
Edit the settings and press Run. Keep this file, dsm_fft.py and dsm_compare.py together."""
import os
import sys

# ---------------- settings ----------------
FILE_A  = "adcfft_withnoise_1151005.txt"      # e.g. with .trannoise
LABEL_A = "with noise"
FILE_B  = "adcfft_nonoise_1151006.txt"        # e.g. without .trannoise
LABEL_B = "no noise"
FS      = 12e6
OSR     = 256
SKIP    = 200
N       = 32768                               # same N / M / SKIP as run_dsm.py
FIN     = 2563.4765625                        # = M * FS / N, M = 7, N = 32768
WINDOW  = "bh"                                # bh | hann
BW      = None                                # None = FS/(2*OSR); e.g. 24300
THRESH  = None                                # None = auto (two-rail voltage) / 0.9
# ------------------------------------------

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)
import dsm_compare

args = [FILE_A, FILE_B, str(FS), str(OSR), "--labels", f"{LABEL_A},{LABEL_B}"]
opts = ["--skip", str(SKIP), "--n", str(N), "--fin", str(FIN), "--window", WINDOW]
if BW is not None:
    opts += ["--bw", str(BW)]
if THRESH is not None:
    opts += ["--thresh", str(THRESH)]
dsm_compare.main(args + opts)
