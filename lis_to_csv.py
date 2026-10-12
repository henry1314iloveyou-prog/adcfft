"""Extract the `.print tran v(qout)` table from an HSPICE .lis file into time,value CSV.

Usage: python lis_to_csv.py run.lis [out.csv]

HSPICE prints numbers with engineering suffixes (62.50000n, 1.8000, 12.34u, ...).
Use `.option numdgt=7` in the netlist so the time column has enough digits.
The output CSV can be given to run_dsm.py directly (THRESH = 0.9).
"""
import re
import sys

SUF = {"t": 1e12, "g": 1e9, "x": 1e6, "meg": 1e6, "k": 1e3, "m": 1e-3,
       "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}
NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|[tgxkmunpfa])?[a-z]*$", re.I)


def hnum(tok):
    m = NUM.match(tok)
    if not m:
        raise ValueError(tok)
    return float(m.group(1)) * SUF.get((m.group(2) or "").lower(), 1.0)


def main(path, out):
    rows, inside, seen_header = [], False, False
    with open(path, errors="replace") as fh:
        for line in fh:
            low = line.strip().lower()
            if not inside:
                # table starts after the header line that contains "time"
                if low.startswith("time") and not seen_header:
                    seen_header = True
                elif seen_header and low and not low.startswith("time"):
                    inside = True
                    # this line is either the signal-name line or already data
                else:
                    continue
            if inside:
                if low.startswith("y") or low.startswith("x") and len(low) <= 2:
                    break                                # end of table marker
                parts = line.split()
                if len(parts) == 2:
                    try:
                        rows.append((hnum(parts[0]), hnum(parts[1])))
                    except ValueError:
                        pass                             # signal-name line, blank, ...
    if not rows:
        sys.exit("no (time, value) rows found; is there a `.print tran v(qout)` in the netlist?")
    with open(out, "w") as fh:
        fh.write("time,v(qout)\n")
        for t, v in rows:
            fh.write(f"{t:.9e},{v:.6e}\n")
    dt = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    print(f"{len(rows)} rows -> {out};  t = {rows[0][0]:.6e} .. {rows[-1][0]:.6e} s;  "
          f"step min/max = {min(dt):.4e} / {max(dt):.4e} s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    main(src, sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + "_qout.csv")
