import numpy as np
import torch
h = torch.LongTensor(np.array([[1,10],[2,20]]))
N = h.size()[0]
print(h)
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
print(a_input)