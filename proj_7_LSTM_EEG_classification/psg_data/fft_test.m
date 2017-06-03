function fft_test(sample, Fs, title_str)

X = sample;
L = length(sample);   % Length of signal

Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title(title_str)
xlabel('f (Hz)')
ylabel('|P1(f)|')

end