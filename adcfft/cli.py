import argparse

from .analyze import analyze, load_bits, plot
from .decimate import decimate


def main(argv=None):
    p = argparse.ArgumentParser(prog="adcfft", description="Delta-sigma ADC spectrum analysis")
    p.add_argument("file", help="capture, one 0/1 value per line")
    p.add_argument("--fs", type=float, required=True, help="modulator rate [Hz]")
    p.add_argument("--bw", type=float, required=True, help="signal bandwidth [Hz]")
    p.add_argument("--decimate", action="store_true",
                   help="CIC+FIR decimate first (BW is used as the filter cutoff)")
    p.add_argument("--window", default="blackmanharris")
    p.add_argument("--plot", metavar="PNG", nargs="?", const="", help="show or save plot")
    a = p.parse_args(argv)

    x = load_bits(a.file)
    fs = a.fs
    if a.decimate:
        x, fs, _ = decimate(x, fs, a.bw)
        x = x[min(200, x.size // 4):]  # drop filter start-up transient
    r = analyze(x, fs, a.bw, window=a.window)
    if a.plot is not None:
        plot(r, a.bw, a.plot or None)


if __name__ == "__main__":
    main()
