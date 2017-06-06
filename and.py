import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

X = torch.Tensor([[0,0], [0, 1], [1, 0], [1,1]])
Y = torch.Tensor([0, 0, 0, 1])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 5)
        self.l2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


net = Net()

input = Variable(X)

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

for i in range(100):
# in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, Variable(Y))
    loss.backward()
    optimizer.step()  
    print(loss.data[0])

print(net(input))
