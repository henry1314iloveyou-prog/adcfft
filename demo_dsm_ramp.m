% DEMO_DSM_RAMP  Ramp input through a 2nd-order 1-bit DSM @ 12 MHz, decimate,
% and report gain/offset error and INL (end-point-free, best-fit line).
% To use real hardware data, replace 'bits' and 'u' (ideal ramp) with your capture.
clear; clc;

fs   = 12e6;               % modulator clock
fc   = 10e3;               % decimator passband edge
Tr   = 50e-3;              % ramp duration [s]
N    = round(Tr*fs);
Vcm  = 0.75;  Vamp = 0.15; % ramp = Vcm +/- Vamp  [V]  (0.6 V -> 0.9 V)
Vfs  = 0.75;               % ASSUMED full scale: modulator +/-1 <-> Vcm +/- Vfs (e.g. VDD = 1.5 V); edit to match your design

Vin = linspace(Vcm-Vamp, Vcm+Vamp, N).';   % ideal ramp [V]
u   = (Vin - Vcm) / Vfs;                   % normalized to +/-1 FS (here -0.2 .. +0.2)

% 2nd-order modulator (same structure as demo_dsm_test.m)
bits = zeros(N,1);  i1 = 0;  i2 = 0;  v = 0;
for n = 1:N
    i1 = i1 + 0.5*(u(n) - v);
    i2 = i2 + 0.5*(i1 - v);
    v  = sign(i2 + (i2==0));
    bits(n) = v;
end

% Decimate: 12 MHz -> /125 -> 96 kHz -> /2 -> 48 kHz
[y, fsOut] = dsm_decimate(bits, fs, fc);

% 16-bit signed output: codes -32768 .. +32767 (1 LSB = Vfs/32768 = 22.9 uV)
NB   = 16;
code = max(min(round(y * 2^(NB-1)), 2^(NB-1)-1), -2^(NB-1));
y    = code / 2^(NB-1);                        % back to FS units for the analysis below

% Align: group delay of CIC (4*(R1-1)/2 samples @fs) + FIR (80 samples @96k)
R1 = 125;  R2 = 2;
dly = (4*(R1-1)/2)/fs + 80/(fs/R1);            % seconds
tOut = (0:numel(y)-1).' / fsOut - dly;
uIdeal = interp1((0:N-1).'/fs, u, tOut, 'linear', NaN);

% Drop filter start-up/edge samples, then best-fit line
ok = ~isnan(uIdeal) & tOut > 2e-3 & tOut < Tr - 2e-3;
ui = uIdeal(ok);  yi = y(ok);
c  = polyfit(ui, yi, 1);
gain = c(1);  offs = c(2);
err  = yi - polyval(c, ui);                    % INL in FS units (output vs fit)

fprintf('Gain = %.5f  (error %.3f %%)\n', gain, 100*(gain-1));
fprintf('Offset = %.3e FS\n', offs);
LSB = 2^-(NB-1);
fprintf('INL peak = %.3e FS = %.1f LSB (16-bit)\n', max(abs(err)), max(abs(err))/LSB);
fprintf('INL rms  = %.3e FS = %.2f LSB\n', std(err), std(err)/LSB);

figure;
subplot(3,1,1); plot((0:N-1)/fs*1e3, u); grid on;
xlabel('t [ms]'); ylabel('input [FS]'); title('Ramp input');
subplot(3,1,2); plot(ui, yi, ui, polyval(c,ui), '--'); grid on;
xlabel('input [FS]'); ylabel('output [FS]  (x32768 = LSB)'); legend('decimated','best fit','Location','best');
subplot(3,1,3); plot(ui, err/LSB); grid on;
xlabel('input [FS]'); ylabel('INL [LSB]'); title('Residual vs best-fit line (includes quantization/idle-tone noise)');
