% DEMO_DSM_RAMP  Ramp input through a 2nd-order 1-bit DSM @ 12 MHz, decimate,
% and report gain/offset error and INL (end-point-free, best-fit line).
% To use real hardware data, replace 'bits' and 'u' (ideal ramp) with your capture.
clear; clc;

fs   = 12e6;               % modulator clock
fc   = 10e3;               % decimator passband edge
Tr   = 4;                 % ramp duration [s]; need >~10 output samples per code for DNL (see below)
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
ok = ~isnan(uIdeal) & tOut > 5e-3 & tOut < Tr - 5e-3;
ui = uIdeal(ok);  yi = y(ok);
c  = polyfit(ui, yi, 1);
gain = c(1);  offs = c(2);
err  = yi - polyval(c, ui);                    % INL in FS units (output vs fit)

fprintf('Gain = %.5f  (error %.3f %%)\n', gain, 100*(gain-1));
fprintf('Offset = %.3e FS\n', offs);
LSB = 2^-(NB-1);
fprintf('INL peak = %.3e FS = %.1f LSB (16-bit)\n', max(abs(err)), max(abs(err))/LSB);
fprintf('INL rms  = %.3e FS = %.2f LSB\n', std(err), std(err)/LSB);

% Smoothed INL: average out wideband noise so only the static nonlinearity remains
errS = movmean(err, 481);                      % ~10 ms @ 48 kHz
fprintf('INL (smoothed) peak = %.2f LSB, noise rms (err - smoothed) = %.2f LSB\n', ...
        max(abs(errS))/LSB, std(err-errS)/LSB);

% ---- DNL / missing codes by code-density (histogram of the ramp) ----
% Only codes the fitted line spans (minus a margin) are evaluated.
cc   = round(yi/LSB);                          % integer output codes
cLo  = min(cc) + 50;  cHi = max(cc) - 50;      % skip edge codes (ramp start/stop)
codes = (cLo:cHi).';
cnt  = accumarray(cc(cc>=cLo & cc<=cHi) - cLo + 1, 1, [numel(codes) 1]);
hitsPerCode = mean(cnt);
DNL  = cnt/hitsPerCode - 1;                    % LSB
INLh = cumsum(DNL);  INLh = INLh - polyval(polyfit(codes, INLh, 1), codes);
nMiss = sum(cnt == 0);
fprintf('Codes evaluated: %d..%d (%d), mean hits/code = %.1f\n', cLo, cHi, numel(codes), hitsPerCode);
fprintf('DNL: min %.2f, max %.2f LSB;  INL (from DNL) peak %.2f LSB\n', min(DNL), max(DNL), max(abs(INLh)));
fprintf('Missing codes (DNL = -1): %d  (%.2f %%)\n', nMiss, 100*nMiss/numel(codes));
if hitsPerCode < 10
    warning('Only %.1f hits/code: DNL/missing-code result is unreliable. Increase Tr.', hitsPerCode);
end
% Note: the decimated output carries >1 LSB of shaped noise, which acts as
% dither and smears code widths, so DNL here is optimistic vs. a noise-free ADC.

figure;
subplot(4,1,1); plot((0:N-1)/fs*1e3, u); grid on;
xlabel('t [ms]'); ylabel('input [FS]'); title('Ramp input');
subplot(4,1,2); plot(ui, yi, ui, polyval(c,ui), '--'); grid on;
xlabel('input [FS]'); ylabel('output [FS]  (x32768 = LSB)'); legend('decimated','best fit','Location','best');
subplot(4,1,3); plot(ui, err/LSB, ui, errS/LSB, 'LineWidth', 1); grid on;
xlabel('input [FS]'); ylabel('INL [LSB]'); title('INL: residual vs best-fit line (raw and smoothed)');
subplot(4,1,4); plot(codes, DNL); grid on;
xlabel('output code'); ylabel('DNL [LSB]'); title(sprintf('DNL (code density), %d missing codes', nMiss));
