import torch
import numpy as np

l1 = np.arange(5)
l2, l3 = l1 * 2, l1 * 3
np.savetxt('test.txt', (l1, l2, l3))
