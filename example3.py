#Broadcasting
import torch
import time
dim=8000

device=torch.device("cuda:0")

start_time = time.time()
torch.add(torch.randn(dim,1), torch.randn(dim))
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
torch.add(torch.randn(dim,1,device=device), torch.randn(dim,device=device))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)