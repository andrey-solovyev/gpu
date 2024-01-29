#Outer Product of tensors
import torch
import time
# dim=8000
#
# device=torch.device("cuda:0")
#
# start_time = time.time()
# torch.outer(torch.randn(dim), torch.randn(dim))
# elapsed_time = time.time() - start_time
# print('CPU_time = ',elapsed_time)
#
# start_time = time.time()
# torch.outer(torch.randn(dim,device=device), torch.randn(dim,device=device))
# elapsed_time = time.time() - start_time
# print('GPU_time = ',elapsed_time)

import random
import torch
import time
import math
# GPU Elapsed time:  22.08850336074829
# CPU Elapsed time:  34.403249979019165
class DynamicNet(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):

        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):

        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'

device = torch.device( 'cuda')

dtype = torch.float
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

model = DynamicNet()


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
start_time = time.time()
for t in range(30000):
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ', elapsed_time)
print(f'Result: {model.string()}')