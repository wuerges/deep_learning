import gym
import matplotlib.pyplot as plt
import time
import numpy as np
#env = gym.make('CartPole-v0')
#print(gym.envs.registry.all())
env = gym.make('FrozenLake-v0')


#Q = np.random((64, 4))
#Q = np.random.random((16,4))
Q = np.zeros((16,4))

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

    a = np.argmax(Q[s,:]  + np.random.randn(1, env.action_space.n)*(1./(episode+1)**2))
    s_, r, done, info = env.step(a)

    if r == 0 and done:
        r = -1
    Q[s,a] += alpha * (r + gamma*np.max(Q[s_,:]) - Q[s,a])


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
