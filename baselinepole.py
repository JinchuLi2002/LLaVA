import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym
import gym.spaces as sp
from tqdm import trange
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from time import sleep

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================
# DQN Components
# ===========================

# Policy Network
class QNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=64):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# DQN Agent
class DQN:
    def __init__(self, n_states, n_actions, batch_size=64, lr=1e-3, gamma=0.99, mem_size=10000, learn_step=5, tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        # model
        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.net_target.load_state_dict(self.net_eval.state_dict())

        self.memory = ReplayBuffer(mem_size, batch_size)
        self.counter = 0

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        # epsilon-greedy
        if random.random() < epsilon:
            return random.choice(np.arange(self.n_actions))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.counter += 1
        if self.counter % self.learn_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)
        q_eval = self.net_eval(states).gather(1, actions)

        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update
        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau)*target_param.data)


def train(env, agent, n_episodes=500, max_steps=200, eps_start=1.0, eps_end=0.05, eps_decay=0.995, target=180, chkpt=False):
    scores = []
    epsilon = eps_start
    pbar = trange(n_episodes)
    for i in pbar:
        state, _ = env.reset()
        score = 0
        for t in range(max_steps):
            action = agent.getAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.save2memory(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        epsilon = max(eps_end, epsilon*eps_decay)
        avg_score = np.mean(scores[-100:])
        pbar.set_postfix_str(f"Episode {i}, Score: {score:.2f}, Avg: {avg_score:.2f}")

        if avg_score >= target and len(scores) >= 100:
            print("Target reached!")
            break

    if chkpt:
        torch.save(agent.net_eval.state_dict(), 'checkpoint_baseline.pth')

    return scores

def testAgent(env, agent, loop=3):
    for i in range(loop):
        state, _ = env.reset()
        score = 0
        for t in range(200):
            action = agent.getAction(state, epsilon=0)
            state, reward, done, _, _ = env.step(action)
            score += reward
            if done:
                print(f"Test Episode {i+1}: Score = {score:.2f}")
                break

def plotScore(scores):
    plt.figure()
    plt.plot(scores)
    plt.title("Score History (No VLM)")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()

def main(args):
    env = gym.make('CartPole-v1')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = DQN(
        n_states=num_states,
        n_actions=num_actions,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        mem_size=args.memory_size,
        learn_step=args.learn_step,
        tau=args.tau
    )

    scores = train(
        env,
        agent,
        n_episodes=args.episodes,
        max_steps=200,
        target=args.target_score,
        chkpt=args.save_chkpt
    )

    plotScore(scores)
    testAgent(env, agent, loop=args.test_loop)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Baseline on CartPole-v1 without VLM")
    # DQN Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DQN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for DQN optimizer")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--target_score", type=float, default=180.0, help="Target average score for early stopping")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for DQN")
    parser.add_argument("--memory_size", type=int, default=5000, help="Replay memory size")
    parser.add_argument("--learn_step", type=int, default=5, help="Frequency of learning steps")
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter for target network")
    parser.add_argument("--save_chkpt", action="store_true", help="Save model checkpoint")
    parser.add_argument("--test_loop", type=int, default=3, help="Number of test episodes")

    args = parser.parse_args()
    main(args)
