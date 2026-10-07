"""Compare the run WITH .trannoise against the run WITHOUT it (N=32768, M=7).
GUI friendly: edit the settings and press Run (F5).
Keep this file, dsm_fft.py and dsm_compare.py together with the data files."""
import os
import sys

# ---------------- settings ----------------
# (file, label, injected SNR = None).  The reference run is the one at REF (-1 = last = without noise).
FILES = [
    ("adcfft_withnoise_1151005.txt", "with .trannoise",    None),
    ("adcfft_nonoise_1151006.txt",   "without .trannoise", None),
]
REF = -1                   # index of the reference run
FS      = 12e6
OSR     = 256
SKIP    = 200
OUT     = "fft_compare_noise_vs_nonoise_32768.png"
N       = 32768            # M=7
FIN     = 2563.4765625
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
