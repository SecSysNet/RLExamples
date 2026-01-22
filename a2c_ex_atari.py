import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical
import gymnasium as gym

## Atari
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84,84))
    frame = frame/255.0 # Normalize pixel value into [0,1]
    return frame

def stack_frames(stacked_frames, new_frame, is_new_episode):
    if is_new_episode:
        stacked_frames = deque([np.zeros((84,84), dtype=np.float32) for _ in range(4)], maxlen=4)
        for _ in range(4):
            stacked_frames.append(new_frame)
        else:
            stacked_frames.append(new_frame)

    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames

class A2CNetwork(nn.Module):
    def __init__(self, input_channels, action_space):
        super(A2CNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # x.shape = [1,64,7,7]
        # 2d -> 1d by x.view(x.size(0), -1)
        self.fc1 = nn.Linear(64*7*7, 512)

        self.actor = nn.Linear(512, action_space)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class A2CAgent:
    def __init__(self, env, network, optimizer, gamma=0.99):
        self.env = env
        self.network = network
        self.gamma = gamma
        self.optimizer = optimizer
        self.stacked_frames = deque(maxlen=4)


    def preprocess_state(self, state, is_new_episode):
        processed_frame = preprocess_frame(state)
        stacked_state, self.stacked_frames = stack_frames(self.stacked_frames, processed_frame, is_new_episode)
        return stacked_state

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.network(state)
        probs = torch.softmax(policy, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), value

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma*R
            returns.insert(0, R)
        return returns


    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state, is_new_episode=True)
            log_probs = []
            values = []
            rewards = []

            done = False
            while not done:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state, is_new_episode=False)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                state = next_state

            returns = self.compute_returns(rewards)

            policy_loss = []
            value_loss = []

            for log_prob, value, R in zip(log_probs, values, returns):
                advantage = R - value.item()
                policy_loss.append(-log_prob*advantage)
                value_loss.append(nn.functional.mse_loss(value, torch.tensor([R])))

            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            loss.backward()
            self.optimizer.step()

            if episode % 50 == 0:
                print(f"Eposode {episode}: Total Reward = {sum(rewards)}")

env = gym.make('ALE/SpaceInvaders-v5', render_mode = 'rgb_array')
#env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

network = A2CNetwork(4, env.action_space.n)
optimizer = optim.Adam(network.parameters(), lr=1e-3)
agent = A2CAgent(env, network, optimizer)

agent.train(num_episodes=500)

def play_trained_agent(env, network, episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        stacked_frames = deque([np.zeros((84,84), dtype=np.float32) for _ in range(4)], maxlen=4)
        stacked_state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True)

        done = False
        total_reward = 0

        while not done:
            env.render()
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0)
            policy, value = network(state_tensor)
            action = torch.argmax(policy, dim=1).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode = False)
            total_reward += reward

        print (f"Episode {episode+1} ended with total reward: {total_reward}")

env=  gym.make("ALE/SpaceInvaders-v5", render_mode='human')
play_trained_agent(env, network)
