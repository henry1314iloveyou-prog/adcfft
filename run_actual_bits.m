% RUN_ACTUAL_BITS  Analyze a real capture (fin = 1 kHz, fs = 12 MHz, OSR = 256).
clear; clc;
fs  = 12e6;
OSR = 256;
bw  = fs/(2*OSR);                    % 23.4375 kHz signal bandwidth
fc  = bw;                            % decimation passband edge

bits = load('adc1khz1150927.txt');   % 61098 samples of 0/1 (CRLF ok)
% Whole number of 1 kHz cycles (12000 samples each): 61098 -> 60000
bits = bits(1 : floor(numel(bits)/12000)*12000);

% 1) Spectrum of the raw 1-bit stream (no decimation needed; only ~5 cycles,
%    so keep every sample -- decimating would leave too few points)
r0 = dsm_analyze(bits, fs, bw);

% 2) Optional: decimate (12 MHz -> /125 -> /2 = 48 kHz) and analyze.
%    Only ~240 output points for 5 cycles, so this is a rough check only.
% [y, fsOut] = dsm_decimate(bits, fs, fc);
% y = y(41:end);                     % drop FIR group delay (80 taps @ 96 kHz -> 40 @ 48 kHz)
% y = y(1 : floor(numel(y)/48)*48);  % 48 samples per 1 kHz cycle at 48 kHz
% r = dsm_analyze(y, fsOut, fc);
