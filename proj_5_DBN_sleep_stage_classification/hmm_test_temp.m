tr = [0.95,0.05;
      0.10,0.90];
          
e = [1/6,  1/6,  1/6,  1/6,  1/6,  1/6;
     1/10, 1/10, 1/10, 1/10, 1/10, 1/2;];

[seq, states] = hmmgenerate(1000,tr,e);
[estimateTR, estimateE] = hmmestimate(seq,states);