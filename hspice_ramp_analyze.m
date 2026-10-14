function r = hspice_ramp_analyze(csvfile, varargin)
%HSPICE_RAMP_ANALYZE  Gain / offset / INL of the 2nd-order DSM from an HSPICE ramp run.
%
%   r = hspice_ramp_analyze('ramp.csv')
%
%   csvfile : numeric CSV with 4 columns  [time  v(qout)  v(vip)  v(vin)]
%             (waveform-viewer export of the .probe signals, header lines allowed)
%   Options (Name,Value):
%     'fs'    12e6      modulator clock
%     'tsamp' 100e-12+0.75/12e6   sampling instant inside each clock (= ts_off in the netlist)
%     'Vth'   0.9       logic threshold of qout [V]
%     'VfsD'  1.5       differential full scale [V]: (vip-vin) = +/-VfsD  <->  +/-1 FS (ASSUMED)
%     'R'     128       sinc^3 decimation ratio (1 ms ramp is too short for dsm_decimate's FIR)
%     'NB'    16        output bits
%
%   A 1 ms ramp gives only ~90 output points: enough for gain/offset/coarse INL,
%   NOT for DNL / missing codes. Wrong VfsD only scales the gain (gain and VfsD are
%   indistinguishable); offset and INL (in LSB of the fitted line) are unaffected.

p = inputParser;
addParameter(p,'fs',12e6);  addParameter(p,'tsamp',100e-12+0.75/12e6);
addParameter(p,'Vth',0.9);  addParameter(p,'VfsD',1.5);
addParameter(p,'R',128);    addParameter(p,'NB',16);
parse(p,varargin{:});  o = p.Results;

d = readmatrix(csvfile);
d = d(all(isfinite(d(:,1:4)),2), 1:4);
[t, iu] = unique(d(:,1));  d = d(iu,:);
tck = 1/o.fs;

% ---- sample qout once per clock, at the same instant as the netlist's ts_off ----
tS   = (o.tsamp : tck : t(end)).';
tS   = tS(tS >= t(1));
bits = 2*(interp1(t, d(:,2), tS) > o.Vth) - 1;          % {-1,+1}
vd   = interp1(t, d(:,3)-d(:,4), tS);                    % differential input [V]
u    = vd / o.VfsD;                                      % [FS]

% ---- sinc^3 decimator (CIC order 3) ----
R = o.R;  K = 3;
h = 1;  for k = 1:K, h = conv(h, ones(R,1)); end
h = h / sum(h);
y  = filter(h, 1, bits);
y  = y(R:R:end);                                         % decimate
ti = tS(R:R:end);
dly = K*(R-1)/2 * tck;                                   % group delay
uy = interp1(tS, u, ti - dly, 'linear', NaN);            % ideal input at the same instant
ok = ~isnan(uy);  ok(1:K+1) = false;                     % drop CIC start-up
uy = uy(ok);  y = y(ok);

% ---- output code, gain / offset / INL ----
LSB  = 2^-(o.NB-1);
code = max(min(round(y/LSB), 2^(o.NB-1)-1), -2^(o.NB-1));
yq   = code*LSB;
c    = polyfit(uy, yq, 1);
err  = yq - polyval(c, uy);

r.gain = c(1);  r.offset = c(2);  r.offset_V = c(2)*o.VfsD;
r.INL_pk_LSB = max(abs(err))/LSB;  r.INL_rms_LSB = std(err)/LSB;
r.npts = numel(uy);
fprintf('Points: %d (R=%d, fout = %.1f kHz)\n', r.npts, R, o.fs/R/1e3);
fprintf('Gain   = %.5f  (%.3f %% from 1, VfsD = %.2f V assumed)\n', c(1), 100*(c(1)-1), o.VfsD);
fprintf('Offset = %.3e FS = %.1f LSB = %.3f mV (input referred, diff)\n', ...
        c(2), c(2)/LSB, 1e3*r.offset_V);
fprintf('INL    = %.1f LSB peak, %.1f LSB rms (includes sinc3 residual noise)\n', ...
        r.INL_pk_LSB, r.INL_rms_LSB);

figure;
subplot(3,1,1); plot(t*1e3, d(:,3), t*1e3, d(:,4)); grid on;
xlabel('t [ms]'); ylabel('V'); legend('vip','vin'); title('Ramp input');
subplot(3,1,2); plot(uy, yq, uy, polyval(c,uy), '--'); grid on;
xlabel('input diff [FS]'); ylabel('output [FS]'); legend('decimated','best fit','Location','best');
subplot(3,1,3); plot(uy, err/LSB); grid on;
xlabel('input diff [FS]'); ylabel('INL [LSB]'); title('Residual vs best-fit line');
end
