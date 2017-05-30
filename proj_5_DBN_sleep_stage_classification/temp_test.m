rng('default')

t = 0:0.001:1-0.001;
x = cos(2*pi*3*t)+randn(size(t));

pband = bandpower(x,128,[0.5 4]);
ptot = bandpower(x,128,[0 64]);
per_power = 100*(pband/ptot);

E(1,:)=[seg2E(x(:,HDR.EEG(1)), width, ss, 'fft') ...    % 1 - 5