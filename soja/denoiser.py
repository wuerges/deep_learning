import numpy as np # linear algebra
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Any results you write to the current directory are saved as output.
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.utils.data as data_utils

import matplotlib
matplotlib.use("TkAgg")


class MLPNet(nn.Module):
    def __init__(self, hs):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, hs)
        self.h = nn.Linear(hs, hs)
        self.fc2 = nn.Linear(hs, 28*28)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.h(x))
        #x = F.relu(self.h(x))
        #x = F.relu(self.h(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 28, 28)
        return x

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 28, 28)
        return x

class L(nn.Module):
    def __init__(self):
        super(L, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.bn2 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(26*26, 10)
        self.sm = nn.Softmax()

    def forward(self, x):
        r = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out.view(-1, 26*26)
        out = self.fc1(out)
        return self.sm(out)


#tt = T.Compose([T.ToTensor(), T.Normalize((0.5,), (1.0,))])
tt = T.ToTensor() #T.Compose([T.ToTensor(), T.Normalize((0.5,), (1.0,))])

ti = T.ToPILImage()

train = dset.MNIST("../mnist", transform=tt)
test =  dset.MNIST("../mnist", transform=tt, train=False)

train_loader = data_utils.DataLoader(dataset=train, batch_size=32)
test_loader = data_utils.DataLoader(dataset=train, batch_size=32)

#net = L().cuda()
net1 = MLPNet(28*28).cuda()
#net2 = MLPNet(1000).cuda()
loss = nn.MSELoss()

opt1 = torch.optim.Adam(net1.parameters())
#opt1 = torch.optim.SGD(net1.parameters(), lr=0.01, momentum=0.9)
#opt2 = torch.optim.SGD(net2.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    for batch, (x, y) in enumerate(train_loader):
        #print(batch, x, y)
        opt1.zero_grad()
        #opt2.zero_grad()
        
        noise = torch.rand(x.size())
        i_x = x + noise

        z1 = net1(Variable(i_x.cuda()))
        #z2 = net2(Variable(x.cuda()))

        l1 = loss(z1, Variable(x.cuda()))
        l1.backward()
        opt1.step()

        #l2 = loss(z2, Variable(z1.data))
        #l2.backward()
        #opt2.step()

        if batch % 500 == 0: 
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch, l1.data[0]))
            #print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch, l2.data[0]))

            fig = plt.figure()
            a = fig.add_subplot(1, 4, 1)
            plt.imshow(ti(x[0]))
            a.set_title("X")
            
            a = fig.add_subplot(1, 4, 2)
            plt.imshow(ti(noise[0]))
            a.set_title("noise")
            
            a = fig.add_subplot(1, 4, 3)
            plt.imshow(ti(i_x[0]))
            a.set_title("X + noise")

            a = fig.add_subplot(1, 4, 4)
            plt.imshow(ti(z1.data[0].cpu().view(1, 28, 28)))
            a.set_title("denoise")

            plt.show()
            #exit(0)
            #plt.imshow(ti(z1.data.cpu()[0]))


#X_train = torch.Tensor([tt(i) for (i, _) in train])

#X_train = [img for img, l in train]
#Y_train = [l for img, l in train]

#print(X_train[:10])

#X = torch.Tensor(tt(x) for x in X_train)
#print("X", X)
#y = net(X)
#print(X, y)


#print(train)

#print(tt(train[0][0]))

#print(train[4])
#print(test[4])

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#x_train = F.from_numpy(x_train)


