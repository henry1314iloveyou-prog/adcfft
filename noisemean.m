clear
%format long;
load meannoise.txt;
ny=size(meannoise,1);
fs=6144000*2                 %   Sampling frequency
Ts=1./fs
t = (0:ny-1)*Ts;
%t=linspace(0,Ts*ny,ny);
y=[t',meannoise];
%y=y';
BW=4000                     %   signal bandwidth
Fn=2*BW;                     %   fn=Nyquist-rate sampling frequency
OSR=64                      %   for decimation filter 

Amp=150e-3;                  %   Amplitude
OP_offset=0e-3               %   OP_offset

VLDOA=1.5
Vref=VLDOA/16*4


%y=y*Vref;
%pts=2.^16;                   %   FFT points
pts=length(y);
%pts=65536

G=2.^-27;                    %   Gain for 1st stage decimation filter
%G=(2.^-27)/8;                    %   Gain for 1st stage decimation filter
Tstop=(pts-1)/fs

fs4=fs./OSR


Ts4=1./fs4
%stem(y);
%pause;
% n=length(y);

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

% 251002 modify,output*2
deci_fir116=round(deci_fir1*2.^24*2);
%G=1;
%deci_fir116=deci_fir1*2.^27/1*2^((G-1)*3)-32768;


%deci_fir116=deci_fir1*2.^16;

deci_fir116(1:5)=[];
z=round(mean(deci_fir116))
plot(deci_fir116);
%hold all

z_amp1=max(deci_fir116)-z
z_amp2=z-min(deci_fir116)

std1=std(deci_fir116)

% % % 
% % Yn=fft(deci_fir1,length(deci_fir1));
% % n=length(Yn);
% % pyy=Yn.*conj(Yn)/((n/2)^2);
% % pyy(((n/2)+1):n)=[];
% % f=fs4.*(0:n-1)/n;
% % f(((n/2)+1):n)=[];
% % Pyy=10.*log10(pyy);
% % semilogx(f,Pyy),grid;
% % %axis([0 56000 -180 0]);
% % title('Power Spectral density')
% % xlabel('Frequency [Hz]')
% % ylabel('[dB]')
% 
% % %sdmout=sdmout((length(sdmout)/2)+1:length(sdmout));
% % 
% y=y';
% y(1:1000)=[];
% nn=length(y);
% %nn=65536;
% nb=1:1:nn;
% 
% kb=0.355768-0.487396*cos(2*pi*nb/(nn))+0.144232*cos(4*pi*nb/(nn))-0.012604*cos(6*pi*nb/(nn));
% 
% yy=y(1:nn);
% yb=kb.*yy;
% 
% Yn=fft(yb,nn);
% n=length(Yn);
% pyy=Yn.*conj(Yn)/((n/2)^2);
% pyy(((n/2)+1):n)=[];
% f=2*fs.*(0:n-1)/n;
% f(((n/2)+1):n)=[];
% Pyy=10.*log10(pyy);
% semilogx(f,Pyy),grid;
% axis([0 5600000 -140 0]);
% title('Power Spectral density')
% xlabel('Frequency [Hz]')
% ylabel('[dB]')