"""Run dsm_fft.py from a Python GUI (IDLE / Spyder / ...): edit the settings, press Run.
Keep this file and dsm_fft.py in the same folder as the bit file."""
import os
import runpy
import sys

# ---------------- settings ----------------
FILE   = "qout_bits.txt"   # one 0/1 per line (qout thresholded at 0.9 V)
FS     = 12e6              # modulator clock [Hz]
OSR    = 256
BW     = None             # signal bandwidth [Hz]; None = FS/(2*OSR) = 23437.5.  e.g. 24300
SKIP   = 200               # start-up samples to drop (= nskip in the netlist)
N      = 32768             # contiguous samples to analyse
FIN    = 2563.4765625      # expected input frequency = M*FS/N, M=7 (N=32768)
WINDOW = "hann"            # hann | bh | rect
CIC_R     = 0              # CIC decimation ratio, 0 = CIC off      (e.g. 128)
CIC_ORDER = 3              # CIC order (>= modulator order + 1)
FIR_R     = 0              # FIR decimation ratio, 0 = FIR off      (e.g. 2)
FIR_FC    = None           # FIR passband edge [Hz]; None = 0.4 * fs_out of the FIR stage
FIR_STOP  = None           # FIR stopband edge [Hz]; None = fs_out - fc
FIR_ATTEN = 70             # FIR stopband attenuation [dB]
CURSOR = True              # interactive cursor line / markers in the plot window
DEC_WINDOW = "bh"          # window for the decimated spectrum (bh recommended)
# both CIC_R and FIR_R = 0 -> only the raw 1-bit spectrum is analysed
THRESH = None              # e.g. 0.9 if FILE is an analog v(qout) export (time, value)
COL = 1                    # CSV column (0-based) holding v(qout); column 0 = time
PHASE = 62.5e-9            # sampling instant inside each clock period [s] = 10p+0.75*tck
IGNORE_TIME = False        # True: rows are exactly one sample per clock; use their order, ignore time column
RESAMPLE = False           # True: re-sample a (time, value) file at 1/FS before thresholding
# ------------------------------------------

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.argv = ["dsm_fft.py", FILE, str(FS), str(OSR), str(CIC_ORDER),
            "--skip", str(SKIP), "--n", str(N), "--fin", str(FIN),
            "--window", WINDOW, "--cic", str(CIC_R), "--fir", str(FIR_R),
            "--fir-atten", str(FIR_ATTEN), "--dec-window", DEC_WINDOW]
if FIR_FC is not None:
    sys.argv += ["--fir-fc", str(FIR_FC)]
if FIR_STOP is not None:
    sys.argv += ["--fir-stop", str(FIR_STOP)]
sys.argv += ["--col", str(COL)]
if BW is not None:
    sys.argv += ["--bw", str(BW)]
if THRESH is not None:
    sys.argv += ["--thresh", str(THRESH)]
sys.argv += ["--cursor", "1" if CURSOR else "0"]
if IGNORE_TIME:
    sys.argv += ["--ignore-time", "1"]
if RESAMPLE:
    sys.argv += ["--resample", "1", "--phase", str(PHASE)]
runpy.run_path("dsm_fft.py", run_name="__main__")
