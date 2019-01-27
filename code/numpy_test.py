import numpy as np


for idx in range(20000):
    if np.mod(idx, 300) == 0:
        print(idx)