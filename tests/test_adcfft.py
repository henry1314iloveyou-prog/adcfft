import numpy as np

from adcfft import analyze, decimate


def dsm2(u):
    y = np.empty(u.size)
    i1 = i2 = v = 0.0
    for n, un in enumerate(u):
        i1 += 0.5 * (un - v)
        i2 += 0.5 * (i1 - v)
        v = 1.0 if i2 >= 0 else -1.0
        y[n] = v
    return y


def test_analyze_second_order():
    fs, n, cyc = 2.048e6, 2 ** 16, 67
    fin = cyc * fs / n
    u = 0.5 * np.sin(2 * np.pi * fin * np.arange(n) / fs)
    r = analyze(dsm2(u), fs, 4e3, verbose=False)
    assert abs(r["fin"] - fin) < fs / n
    assert abs(r["Asig_dBFS"] - 20 * np.log10(0.5)) < 1
    assert r["SNR"] > 60


def test_decimate_rate():
    y, fso, h = decimate(np.random.randint(0, 2, 12000 * 5), 12e6, 10e3)
    assert fso == 48e3 and y.size == 12000 * 5 // 250
    assert abs(h["cic"].sum() - 1) < 1e-12
