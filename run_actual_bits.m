% RUN_ACTUAL_BITS  Decimate + FFT-analyze a real capture (fin = 1 kHz, fs = 12 MHz).
clear; clc;
fs = 12e6;  fc = 10e3;  fin = 1e3;

bits = load('capture.txt');          % <-- your 0/1 stream, one value per line
% Use a whole number of 1 kHz cycles (12000 samples each) for a clean FFT
bits = bits(1 : floor(numel(bits)/12000)*12000);

% 1) Decimate in MATLAB (CIC /125 -> LPF -> /2  =>  48 kHz)
[y, fsOut] = dsm_decimate(bits, fs, fc);

% Optional: same filter chain in Simulink (results should match)
% build_dsm_decim_model(bits(1:12000*100), fs, fc);   % 100 ms; y2 = decimOut.Data

% 2) Drop filter start-up transient, keep power-of-2-independent whole cycles
y = y(200:end);
y = y(1 : floor(numel(y)/48)*48);    % 48 samples per 1 kHz cycle at 48 kHz

% 3) FFT analysis of the decimated output (BW = fc)
r = dsm_analyze(y, fsOut, fc);

% (Also analyze the raw 1-bit stream at full rate)
% r0 = dsm_analyze(bits, fs, fc);
