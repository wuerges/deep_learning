import gym
import matplotlib.pyplot as plt
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#env = gym.make('CartPole-v0')
#print(gym.envs.registry.all())

X = torch.Tensor([[0,0], [0, 1], [1, 0], [1,1]])
Y = torch.Tensor([0, 0, 0, 1])

net = nn.Sequential(nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Linear(16, 4),
                    nn.ReLU())

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

env = gym.make('FrozenLake-v0')

#Q = np.random((64, 4))
#Q = np.random.random((16,4))
#Q = np.zeros((16,4))

state = 0
win = 0
tries = 1

alpha = 0.21
gamma = 0.96

s = env.reset()

reward_t = 0
rewards = []

log = 100 * [0]

for episode in range(500000):
    optimizer.zero_grad()   # zero the gradient buffers
    print(s)
    x = torch.Tensor(s)
    output = net(Variable(torch.Tensor([float(s)])))

    #env.render()
    a = np.argmax(output  + np.random.randn(1, env.action_space.n)*(1./(episode+1)**2))
    s_, r, done, info = env.step(a)

    if r == 0 and done:
        r = -1
    
    diff = output
    diff[a] = r + gamma*np.max(net(Variable(s_.reshape(1, 1))))
    
    loss = criterion(output, Variable(diff))
    loss.backward()
    optimizer.step()
    #Q[s,a] += alpha * (r + gamma*np.max(Q[s_,:]) - Q[s,a])


    reward_t += r
    s = s_


    if done:
        print("win%:", sum(log[-100:])/100)
        rewards.append(reward_t)
        if reward_t > 0:
            log.append(1)
            win += 1
        else:
            log.append(0)
        tries += 1
        reward_t = 0
        env.reset()
