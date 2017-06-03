import numpy as np

num_samples = 100
perm = np.arange(num_samples)
np.random.shuffle(perm)

print(perm)

batch_size = 5

print(perm[:batch_size])
