function r = dsm_analyze(bits, fs, bw, varargin)
%DSM_ANALYZE  Spectral test of a 1-bit delta-sigma ADC output stream.
%
%   r = dsm_analyze(bits, fs, bw)
%   r = dsm_analyze(bits, fs, bw, 'Name', Value, ...)
%
%   bits : vector of modulator output samples, either {0,1} or {-1,+1}
%   fs   : modulator sampling rate [Hz]
%   bw   : signal bandwidth [Hz]  (OSR = fs / (2*bw))
%
%   Options:
%     'Window'    'blackmanharris' (default) | 'hann' | 'rect'
%     'NHarm'     number of harmonics counted in THD (default 5)
%     'DCBins'    bins near DC excluded from noise (default 3)
%     'Plot'      true (default) | false
%     'FullScale' full-scale amplitude of the +/-1 stream (default 1)
%
%   Output struct r: fin, Asig_dBFS, SNR, SINAD, SFDR, THD, ENOB, OSR, f, PdB
%   No toolboxes required.

p = inputParser;
addParameter(p, 'Window', 'blackmanharris');
addParameter(p, 'NHarm', 5);
addParameter(p, 'DCBins', 3);
addParameter(p, 'Plot', true);
addParameter(p, 'FullScale', 1);
parse(p, varargin{:});
o = p.Results;

x = double(bits(:));
if all(x == 0 | x == 1)
    x = 2*x - 1;                          % map {0,1} -> {-1,+1}
end
x = x / o.FullScale;
N = numel(x);
x = x - mean(x);                          % remove DC

% ---- window ----
n = (0:N-1).';
switch lower(o.Window)
    case 'rect'
        w = ones(N,1);  hw = 1;
    case 'hann'
        w = 0.5 - 0.5*cos(2*pi*n/N);  hw = 3;
    otherwise
        w = 0.35875 - 0.48829*cos(2*pi*n/N) + 0.14128*cos(4*pi*n/N) ...
            - 0.01168*cos(6*pi*n/N);  hw = 5;
end

% ---- spectrum (one-sided power, bins 0..N/2) ----
X  = fft(x .* w);
nb = floor(N/2) + 1;
P  = abs(X(1:nb)).^2;
P(2:end-1) = 2*P(2:end-1);
P  = P / sum(w)^2 / 2;                    % sine of amplitude A -> A^2/2 total (scaled below)
f  = (0:nb-1).' * fs / N;
bwBin = min(nb, floor(bw / (fs/N)) + 1);  % last in-band bin (1-based index)

% ---- locate fundamental in-band ----
inb = (o.DCBins+2):bwBin;
[~, k] = max(P(inb));
k0 = inb(k);                              % 1-based bin index of fundamental

sigIdx = @(kc) max(1,kc-hw) : min(nb,kc+hw);
Psig = sum(P(sigIdx(k0)));

% ---- harmonics (aliased into Nyquist zone) ----
hIdx = [];  Ph = 0;
for h = 2:o.NHarm+1
    kh = mod((k0-1)*h, N);
    if kh > N/2, kh = N - kh; end
    kh = kh + 1;
    if kh <= bwBin && kh > k0+2*hw
        idx = sigIdx(kh);
        hIdx = [hIdx idx];                %#ok<AGROW>
        Ph = Ph + sum(P(idx));
    end
end
hIdx = unique(hIdx);

% ---- noise ----
mask = false(nb,1);
mask(o.DCBins+2:bwBin) = true;            % in-band, excluding DC leakage
mask(sigIdx(k0)) = false;
Pn_noharm = sum(P(mask & ~ismember((1:nb).', hIdx)));
Pn_all    = sum(P(mask));

SNR   = 10*log10(Psig / Pn_noharm);
SINAD = 10*log10(Psig / Pn_all);
THD   = 10*log10(max(Ph,eps) / Psig);
tmp = P;  tmp(~mask) = 0;                 % worst in-band spur
SFDR  = 10*log10(Psig / max(tmp));
ENOB  = (SINAD - 1.76) / 6.02;

% amplitude in dBFS: sine of amplitude A has power A^2/2 (P scaled so FS sine = 0.5)
Asig_dBFS = 10*log10(2*Psig);

r = struct('fin', f(k0), 'Asig_dBFS', Asig_dBFS, 'SNR', SNR, 'SINAD', SINAD, ...
           'SFDR', SFDR, 'THD', THD, 'ENOB', ENOB, 'OSR', fs/(2*bw), ...
           'f', f, 'PdB', 10*log10(P + eps));   % PdB: bin power, 0 dB = 0.5 (FS sine)

fprintf('N = %d, fs = %g Hz, BW = %g Hz, OSR = %g\n', N, fs, bw, r.OSR);
fprintf('Fundamental : %.4f Hz  (%.2f dBFS)\n', r.fin, r.Asig_dBFS);
fprintf('SNR         : %.2f dB\n', SNR);
fprintf('SINAD       : %.2f dB\n', SINAD);
fprintf('THD         : %.2f dB\n', THD);
fprintf('SFDR        : %.2f dB\n', SFDR);
fprintf('ENOB        : %.2f bits\n', ENOB);

if o.Plot
    figure('Name','Delta-sigma ADC spectrum');
    semilogx(f(2:end), r.PdB(2:end) + 10*log10(2)); hold on;
    xline(bw, '--r', 'BW');
    grid on; xlabel('Frequency [Hz]'); ylabel('Power [dB, rel. full-scale sine]');
    title(sprintf('SNR %.1f dB, SINAD %.1f dB, ENOB %.2f', SNR, SINAD, ENOB));
    hold off;
end
end
