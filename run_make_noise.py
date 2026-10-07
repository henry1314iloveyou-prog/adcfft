"""Generate the Gaussian-noise PWL include files for HSPICE -- IDLE friendly.
Edit the settings and press Run (F5). Keep this file and make_noise_pwl.py in one folder.
The .inc files are written to this folder; put them next to your netlist."""
import os
import subprocess
import sys

# ---------------- settings ----------------
SNR_LIST = [90, 80, 70]    # in-band noise-only SNR to inject [dB]; one .inc file per value
N_POINTS = 17000           # >= N + nskip + a few  (N=16384 -> 17000;  N=32768 -> 33000)
SEED     = 1               # same seed -> identical files every time
FS       = 12e6
SIGNAL_POWER = 0.045       # V^2 = (differential peak 0.30 V)^2 / 2
BW       = 23437.5         # Hz
NODE_P, NODE_N, NAME = "vip", "vip_s", "vnp"   # series source: V(vip) = V(vip_s) + noise
# ------------------------------------------

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
for snr in SNR_LIST:
    out = f"noise_{snr}dB.inc"
    subprocess.run([sys.executable, "make_noise_pwl.py", "--snr", str(snr), "--n", str(N_POINTS),
                    "--seed", str(SEED), "--fs", str(FS), "--ps", str(SIGNAL_POWER),
                    "--bw", str(BW), "--out", out, "--name", NAME, "--np", NODE_P, "--nn", NODE_N],
                   check=True)
print("done. Files are in:", here)
