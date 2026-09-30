clear
%format long;
load adc1khz1150927.txt;
ny=size(adc1khz1150927, 1);
fs=12000000                 %   Sampling frequency
Ts=1./fs
t=linspace(0,Ts*ny,ny);
y=[t',adc1khz1150927];
%y=y';
BW=10000                     %   signal bandwidth
Fn=2*BW;                     %   fn=Nyquist-rate sampling frequency
OSR=64                  %   for decimation filter 

Amp=100e-3;                  %   Amplitude
OP_offset=0e-3               %   OP_offset
Vref=0.6
%y=y*Vref;
%pts=2.^16;                   %   FFT points
pts=length(y);
%pts=8192

%G=2.^1;                    %   Gain for 1st stage decimation filter
G=2.^-27;                    %   Gain for 1st stage decimation filter
Tstop=(pts-1)/fs

fs4=fs./OSR


Ts4=1./fs4
%stem(y);
%pause;
 n=length(y);
% 
% for i=1:n
% 
%         if y(i)>0
%             y(i)=1;
%         else
%             y(i)=-1;
%         end
% end

%y=y';
open_system('deci_hspice')
sim('deci_hspice');


%pause;

% deci_fir=deci_fir1((length(deci_fir1)/2)+1:length(deci_fir1));
% 
deci_fir116=deci_fir1*2.^24;
deci_fir116(1:50)=[];
plot(deci_fir116)
[ymax,idmax]=max(deci_fir116)
hold on
plot(idmax, ymax, '^r')
txt=[num2str(ymax)];
%str='ymax';
text(idmax, ymax, txt)
%text(idmax, ymax, str)
hold off
[ymin,idmin]=min(deci_fir116)
hold on
plot(idmin, ymin, 'Ok')
%str='min';
txt=[num2str(ymin)];
text(idmin, ymin, txt)
hold off
%ymean=mean(deci_fir116)
%yline(ymean)
grid on
title('adc1khzwithnoise')
xlabel('Time [s]')
ylabel('Count')


decimean = mean(deci_fir116);
decimax = max(deci_fir116);
decimin = min(deci_fir116);
maxcount = decimax - decimin;

% --- Time Domain Plot ---
figure;
plot(deci_fir116);
grid on;
title('Decimated Output - Time Domain');
xlabel('Sample Index');
ylabel('Count');

% --- FFT and Power Spectral Density with Blackman-Harris & Zero-Padding ---
N_signal = length(deci_fir116);

% 1. Apply a Blackman-Harris window (superior sidelobe suppression)
w = blackmanharris(N_signal);
signal_windowed = deci_fir116 .* w;

% 2. Zero-pad to a large FFT size (e.g., 65536 points) for a smooth, high-resolution curve
N_fft = max(2^16, 4 * nextpow2(N_signal)); 

% 3. Compute FFT with zero-padding
Yn = fft(signal_windowed, N_fft);

% 4. Compute power spectrum with proper window scaling
pyy = (Yn .* conj(Yn)) / (sum(w)^2); 
pyy(2:end-1) = 2 * pyy(2:end-1); % Single-sided spectrum scaling
pyy(floor(N_fft/2)+2:end) = [];

% 5. Normalize to dBFS (Full Scale)
FullScale = max(abs(deci_fir116)); 
pyy_dbfs = pyy / (FullScale^2);
Pyy_dB = 10 * log10(pyy_dbfs + eps);

% 6. Frequency vector for the expanded FFT size
f = fs4 * (0:floor(N_fft/2)) / N_fft;

% Exclude DC (0 Hz) for the semilog plot
f_plot = f(2:end);
Pyy_plot = Pyy_dB(2:end);

% 7. Plotting
figure;
semilogx(f_plot, Pyy_plot, 'LineWidth', 1.2);
grid on;
title('Power Spectral Density (Blackman-Harris Window, High-Resolution)');
xlabel('Frequency [Hz]');
ylabel('Power [dBFS]');

% Zoom in around the lower frequencies to inspect the 1 kHz tone
xlim([10 fs4/2]); 
ylim([-180 10]);

% --- DIAGNOSTIC SCRIPT: Compare Raw vs Decimated Spectra ---

% 1. Check Raw Input Spectrum
N_raw = length(adc1khz1150927);
w_raw = hann(N_raw);
Y_raw = fft(adc1khz1150927 .* w_raw);
P_raw = 10 * log10((Y_raw .* conj(Y_raw)) / (sum(w_raw)^2) + eps);
f_raw = fs * (0:floor(N_raw/2)) / N_raw;
P_raw_single = P_raw(1:floor(N_raw/2)+1);
P_raw_single(2:end-1) = 2 * P_raw_single(2:end-1);

figure('Name', 'Diagnostics');
subplot(2,1,1);
plot(f_raw(2:end), P_raw_single(2:end), 'r', 'LineWidth', 1);
grid on;
title('1. RAW Input Data Spectrum (Before Decimation)');
xlabel('Frequency [Hz]');
ylabel('Power [dB]');
xlim([0 10000]); % Zoom on 1kHz

% 2. Check Decimated Filter Output Spectrum
N_dec = length(deci_fir116);
w_dec = hann(N_dec);
Y_dec = fft(deci_fir116 .* w_dec);
P_dec = 10 * log10((Y_dec .* conj(Y_dec)) / (sum(w_dec)^2) + eps);
f_dec = fs4 * (0:floor(N_dec/2)) / N_dec;
P_dec_single = P_dec(1:floor(N_dec/2)+1);
P_dec_single(2:end-1) = 2 * P_dec_single(2:end-1);

subplot(2,1,2);
plot(f_dec(2:end), P_dec_single(2:end), 'b', 'LineWidth', 1);
grid on;
title('2. DECIMATED Output Spectrum (After deci_hspice)');
xlabel('Frequency [Hz]');
ylabel('Power [dB]');
xlim([0 10000]); % Zoom on 1kHz
disp([max(deci_fir116), min(deci_fir116), mean(deci_fir116)]);

