function [y, fsOut, h] = dsm_decimate(bits, fs, fc, varargin)
%DSM_DECIMATE  Decimate a 1-bit delta-sigma stream (CIC + lowpass FIR).
%
%   [y, fsOut, h] = dsm_decimate(bits, fs, fc)
%
%   bits : {0,1} or {-1,+1} stream at rate fs [Hz]
%   fc   : lowpass cutoff (passband edge) [Hz], e.g. 10e3
%   Options: 'R1' CIC decimation (default 125), 'R2' FIR decimation (default 2),
%            'CICOrder' (default 4, use >= modulator order + 1),
%            'StopEdge' FIR stopband edge [Hz] (default 1.4*fc), 'NTaps' (default 161)
%
%   Defaults for fs = 12 MHz: 12 MHz -> /125 -> 96 kHz -> /2 -> 48 kHz.
%   h.cic and h.lp hold the filter coefficients (same ones the Simulink
%   model built by build_dsm_decim_model.m uses). Needs Signal Processing Toolbox (firpm).

p = inputParser;
addParameter(p, 'R1', 125);
addParameter(p, 'R2', 2);
addParameter(p, 'CICOrder', 4);
addParameter(p, 'StopEdge', 1.4*fc);
addParameter(p, 'NTaps', 161);
parse(p, varargin{:});
o = p.Results;

x = double(bits(:));
if all(x == 0 | x == 1), x = 2*x - 1; end

h = dsm_decim_coeffs(fs, fc, o);

% Stage 1: CIC (= cascade of boxcars, as one FIR) then downsample
v = filter(h.cic, 1, x);
v = v(1:o.R1:end);
fs1 = fs / o.R1;
% Stage 2: lowpass FIR then downsample
v = filter(h.lp, 1, v);
y = v(1:o.R2:end);
fsOut = fs1 / o.R2;
end

function h = dsm_decim_coeffs(fs, fc, o)
box = ones(o.R1,1);
cic = 1;
for k = 1:o.CICOrder, cic = conv(cic, box); end
h.cic = cic / sum(cic);                       % unity DC gain
fs1 = fs / o.R1;
h.lp = firpm(o.NTaps-1, [0 fc o.StopEdge fs1/2]/(fs1/2), [1 1 0 0], [1 5]).';
end
