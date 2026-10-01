"""Run dsm_fft.py from a Python GUI (IDLE / Spyder / ...): edit the settings, press Run.
Keep this file and dsm_fft.py in the same folder as the bit file."""
import os
import runpy
import sys

# ---------------- settings ----------------
FILE   = "qout_bits.txt"   # one 0/1 per line (qout thresholded at 0.9 V)
FS     = 12e6              # modulator clock [Hz]
OSR    = 256
SKIP   = 200               # start-up samples to drop (= nskip in the netlist)
N      = 16384             # contiguous samples to analyse
FIN    = 3662.109375       # expected input frequency = M*FS/N, M=5
WINDOW = "hann"            # hann | bh | rect
USE_CIC = False            # False: analyse the raw bit stream
# ------------------------------------------

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.argv = ["dsm_fft.py", FILE, str(FS), str(OSR), "3",
            "--skip", str(SKIP), "--n", str(N), "--fin", str(FIN),
            "--window", WINDOW, "--nocic", "0" if USE_CIC else "1"]
runpy.run_path("dsm_fft.py", run_name="__main__")
