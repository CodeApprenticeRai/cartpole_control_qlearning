import gym
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

'''
*Discretize the state space
*Initialize Value function estimate
*iteratively inform value function estimate until target update reached
'''

env = gym.make('CartPole-v0')
env.reset()


'''
In order to discretize the state space I run a few sample episodes
to inform a distribution of which states are experienced. 

Estimate a neural network transfer function
'''
num_episodes=20

X = []
Y = []
Z = []
A = []
wrapper = [X,Y,Z,A]
pdfs = [[] for i in range(len(wrapper))]

for episode_number in range(num_episodes):
    observation = env.reset()
    done = False
    for t in range(100):
        print(observation, done, sep=" ")
        for i in range(len(wrapper)):
            wrapper[i].append(observation[i])

        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)


env.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(wrapper[0],
                 wrapper[1],
                 wrapper[2],
                 c=wrapper[3],
                 cmap=plt.hot())
fig.colorbar(img)
plt.show()