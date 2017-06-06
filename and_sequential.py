import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

X = torch.Tensor([[0,0], [0, 1], [1, 0], [1,1]])
Y = torch.Tensor([0, 0, 0, 1])

net = nn.Sequential(nn.Linear(2, 5),
                    nn.ReLU(),
                    nn.Linear(5, 1),
                    nn.ReLU()
                    )

input = Variable(X)

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

for i in range(1000):
# in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, Variable(Y))
    loss.backward()
    optimizer.step()  
    print(loss.data[0])

print(net(input))
