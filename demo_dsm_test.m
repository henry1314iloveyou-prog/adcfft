% DEMO_DSM_TEST  Generate a 2nd-order 1-bit delta-sigma stream and analyze it.
% To test real hardware data, replace 'bits' with your capture, e.g.
%   bits = load('capture.txt');   % one 0/1 value per line
clear; clc;

fs  = 2.048e6;             % modulator clock
bw  = 4e3;                 % signal bandwidth  (OSR = 256)
N   = 2^16;                % record length
cyc = 67;                  % odd number of cycles -> coherent sampling
fin = cyc * fs / N;
A   = 0.5;                 % input amplitude (fraction of full scale)

t = (0:N-1)' / fs;
u = A * sin(2*pi*fin*t);

% 2nd-order modulator (Boser-Wooley style)
y = zeros(N,1);  i1 = 0;  i2 = 0;  v = 0;
for n = 1:N
    i1 = i1 + 0.5*(u(n) - v);
    i2 = i2 + 0.5*(i1 - v);
    v  = sign(i2 + (i2==0));
    y(n) = v;
end

r = dsm_analyze(y, fs, bw);
