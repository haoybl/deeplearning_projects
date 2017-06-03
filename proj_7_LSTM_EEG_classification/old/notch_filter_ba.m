function [b, a] = notch_filter_ba(fs, f0)
%% Notch Filter at Given Frequency -- Returns filter coefficients b & a
% Given Input sequence at sampling frequency Fs, notch filter is designed
% at notch frequency f0 to remove power line interference.
% Original Source:
% http://dsp.stackexchange.com/questions/1088/filtering-50hz-using-a-notch-filter-in-matlab

fn = fs/2;              %#Nyquist frequency
freqRatio = f0/fn;      %#ratio of notch freq. to Nyquist freq.

notchWidth = 0.1;       %#width of the notch

%Compute zeros
notchZeros = [exp( sqrt(-1)*pi*freqRatio ), exp( -sqrt(-1)*pi*freqRatio )];

%#Compute poles
notchPoles = (1-notchWidth) * notchZeros;

%figure;
%zplane(notchZeros.', notchPoles.');

b = poly( notchZeros ); %# Get moving average filter coefficients
a = poly( notchPoles ); %# Get autoregressive filter coefficients

%figure;
%freqz(b,a,32000,fs)

%#filter signal x
%output = filter(b,a,input);