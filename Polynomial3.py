import torch
import math
import time
# device = torch.device("cpu")
torch.set_num_threads(6)
# Elapsed time:  3.5520360469818115
device = torch.device("cuda") # Uncomment this to run on GPU
print('Using device:', device)
print()

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()).to(device))
        self.b = torch.nn.Parameter(torch.randn(()).to(device))
        self.c = torch.nn.Parameter(torch.randn(()).to(device))
        self.d = torch.nn.Parameter(torch.randn(()).to(device))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


# Construct our model by instantiating the class defined above
model = Polynomial3()
model.to(device)
print(next(model.parameters()).is_cuda) # True
x = torch.linspace(-math.pi, math.pi, 2000, device=device)
y = torch.sin(x)
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the nn.Linear
# module which is members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
start_time = time.time()
for t in range(50000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 10000 == 0:
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', torch.cuda.memory_allocated(device=device) , 'GB')
            print('Cached:   ', torch.cuda.memory_reserved(device=device) , 'GB')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ', elapsed_time)
print(f'Result: {model.string()}')

print('\n\nControl Flow + Weight Sharing')
