import torch
import torch.nn as nn
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
t1 = torch.randn(100)
t2 = torch.randn(100).to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t2)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')
t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t1.is_cuda) # False
t1 = t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]], device='cuda:0')
print(t1.is_cuda) # True

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)

    def forward(self, x):
        x = self.l1(x)
        return x
model = M()   # not on cuda
model.to(dev) # is on cuda (all parameters)
if dev.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', torch.cuda.memory_allocated(device=dev), 'GB')
    print('Cached:   ', torch.cuda.memory_reserved(device=dev), 'GB')
    print('Cached:   ', torch.cuda.memory_usage(device=dev), 'GB')


print(next(model.parameters()).is_cuda) # True