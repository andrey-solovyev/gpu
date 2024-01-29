import time

import torch

dim=8000

start_time = time.time()
x=torch.randn(dim,dim)
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
x=torch.randn((dim,dim), device=torch.device("cuda:0"))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)