import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import gym
import gym.spaces as sp
from tqdm import trange
from time import sleep
from collections import namedtuple, deque
import matplotlib.pyplot as plt

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer

import os
import imageio
from PIL import Image, ImageDraw, ImageFont

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================
# RewardFunction Class
# ===========================

class RewardFunction:
    def __init__(self, args):
        disable_torch_init()
        self.args = args
        self.model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path,
            args.model_base,
            self.model_name,
            args.load_8bit,
            args.load_4bit,
            device=args.device
        )
        
        # Determine conversation mode
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        self.conv = conv_templates[self.args.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles

        self.image_tensor = None
        self.image_size = None

    def get_reward(self, image):
        """
        Takes a PIL Image, processes it through the llava model,
        and returns a float reward.
        """
        self.image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        self.image_tensor = image_tensor

        # Prepare prompt with image tokens
        prompt = 'describe the image'
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        # Reset conversation
        self.conv = conv_templates[self.args.conv_mode].copy()

        self.conv.append_message(self.roles[0], prompt)
        self.conv.append_message(self.roles[1], None)
        full_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.model.device)

        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        print(f"Input IDs shape: {input_ids.shape}")  # Debug Statement

        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.image_tensor,
                image_sizes=[self.image_size],
                do_sample=False,  # deterministic
                max_new_tokens=100,
                streamer=streamer,
                use_cache=True
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        self.conv.messages[-1][-1] = output_text
        print(output_text)
        # Extract float from the output_text
        try:
            txt = output_text.split()[0]
            if txt[-4:] == "USER": txt = txt[:-4]
            # Assuming the model outputs something like "0.85"
            reward = float(txt)
            reward = max(0.0, min(1.0, reward))  # Clamp between 0 and 1
        except:
            # If parsing fails, assign a default small reward
            reward = 0.0
            print(f"[Warning] Failed to parse reward from model output: '{output_text}'. Assigning reward=0.0")

            print(f"Reward: {reward}")  # Debug Statement

        return reward

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
class ReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen = memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

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

# DQN Agent
class DQN():
    def __init__(self, n_states, n_actions, reward_fn, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau
        self.reward_fn = reward_fn

        # model
        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Initialize target network
        self.net_target.load_state_dict(self.net_eval.state_dict())

        # memory
        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0    # update cycle counter

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action

    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)          # target, if terminal then y_j = rewards
        q_eval = self.net_eval(states).gather(1, actions)

        # loss backprop
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)

# ===========================
# Training and Testing Functions
# ===========================

def train(env, agent, reward_fn, n_episodes=2000, max_steps=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.995, target=200, chkpt=False):
    score_hist = []
    epsilon = eps_start

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    for idx_epi in pbar:
        # Update env.reset() to handle tuple return
        reset_result = env.reset()
        if isinstance(reset_result, tuple) or isinstance(reset_result, list):
            state, _ = reset_result
        else:
            state = reset_result

        score = 0
        for idx_step in range(max_steps):
            action = agent.getAction(state, epsilon)
            step_result = env.step(action)
            
            # Handle different Gym versions
            if len(step_result) == 4:
                next_state, _, done, _ = step_result
            elif len(step_result) == 5:
                next_state, _, done, truncated, _ = step_result
                done = done or truncated
            else:
                raise ValueError("Unexpected number of return values from env.step()")

            # Get rendered image
            image = env.render()
            pil_image = Image.fromarray(image).convert('RGB')

            # Get reward from the RewardFunction
            reward = reward_fn.get_reward(pil_image)
            agent.save2memory(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        score_hist.append(score)
        score_avg = np.mean(score_hist[-100:])
        epsilon = max(eps_end, epsilon*eps_decay)

        pbar.set_postfix_str(f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}")
        pbar.update(0)

        # Early stop
        if len(score_hist) >= 100:
            if score_avg >= target:
                pbar.close()
                print("\nTarget Reached!")
                break

    else:
        print("\nDone!")

    if chkpt:
        torch.save(agent.net_eval.state_dict(), 'checkpoint.pth')

    return score_hist

def testLander(env, agent, reward_fn, loop=3):
    for i in range(loop):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) or isinstance(reset_result, list):
            state, _ = reset_result
        else:
            state = reset_result

        score = 0
        for idx_step in range(500):
            action = agent.getAction(state, epsilon=0)
            env.render()
            # Get rendered image
            image = env.render()
            pil_image = Image.fromarray(image).convert('RGB')

            # Get reward from the RewardFunction
            reward = reward_fn.get_reward(pil_image)
            step_result = env.step(action)

            if len(step_result) == 4:
                state, _, done, _ = step_result
            elif len(step_result) == 5:
                state, _, done, truncated, _ = step_result
                done = done or truncated
            else:
                raise ValueError("Unexpected number of return values from env.step()")

            score += reward
            if done:
                print(f"Test Episode {i+1}: Score = {score:.2f}")
                break
    env.close()

def plotScore(scores):
    plt.figure()
    plt.plot(scores)
    plt.title("Score History")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()

# Functions to save GIFs
def TextOnImg(img, score):
    img = Image.fromarray(img)
    try:
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 18)
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", font=font, fill=(255, 255, 255))

    return np.array(img)

def save_frames_as_gif(frames, filename, path="gifs/"):
    if not os.path.exists(path):
        os.makedirs(path)
        
    print("Saving gif...", end="")
    imageio.mimsave(os.path.join(path, filename + ".gif"), frames, fps=60)

    print("Done!")

def gym2gif(env, agent, reward_fn, filename="gym_animation", loop=3):
    frames = []
    for i in range(loop):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) or isinstance(reset_result, list):
            state, _ = reset_result
        else:
            state = reset_result

        score = 0
        for idx_step in range(500):
            frame = env.render()
            frames.append(TextOnImg(frame, score))
            action = agent.getAction(state, epsilon=0)
            # Get rendered image
            pil_image = Image.fromarray(frame).convert('RGB')
            # Get reward from the RewardFunction
            reward = reward_fn.get_reward(pil_image)
            step_result = env.step(action)

            if len(step_result) == 4:
                state, _, done, _ = step_result
            elif len(step_result) == 5:
                state, _, done, truncated, _ = step_result
                done = done or truncated
            else:
                raise ValueError("Unexpected number of return values from env.step()")

            score += reward
            if done:
                break
    env.close()
    save_frames_as_gif(frames, filename=filename)

# ===========================
# Main Function
# ===========================

def main(args):
    # Confirm HF_HOME
    print(f"HF_HOME is set to: {os.getenv('HF_HOME')}")

    # Initialize environment with render_mode='rgb_array'
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize RewardFunction
    reward_fn = RewardFunction(args)

    # Initialize DQN agent
    agent = DQN(
        n_states = num_states,
        n_actions = num_actions,
        reward_fn = reward_fn,
        batch_size = args.batch_size,
        lr = args.lr,
        gamma = args.gamma,
        mem_size = args.memory_size,
        learn_step = args.learn_step,
        tau = args.tau,
        )

    # Train the agent
    score_hist = train(
        env, 
        agent, 
        reward_fn, 
        n_episodes=args.episodes, 
        target=args.target_score, 
        chkpt=args.save_chkpt
        )

    # Plot the scores
    plotScore(score_hist)

    # Test the agent
    testLander(env, agent, reward_fn, loop=args.test_loop)

    # Save GIF
    if args.save_gif:
        gym2gif(env, agent, reward_fn, filename=args.gif_filename, loop=args.gif_loop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

# ===========================
# Argument Parser
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN with LLaVA-based Reward Function for LunarLander-v2")
    
    # LLaVA Model Arguments
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b", help="Path to the pretrained LLaVA model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name if different from model path")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode for LLaVA")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load the model on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for LLaVA")

    # DQN Training Arguments
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DQN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for DQN optimizer")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--target_score", type=float, default=250.0, help="Target average score for early stopping")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for DQN")
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay memory size")
    parser.add_argument("--learn_step", type=int, default=5, help="Frequency of learning steps")
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter for target network")
    parser.add_argument("--save_chkpt", action="store_true", help="Save model checkpoint")
    
    # Testing and GIF Arguments
    parser.add_argument("--test_loop", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--save_gif", action="store_true", help="Save test episodes as GIF")
    parser.add_argument("--gif_filename", type=str, default="gym_animation", help="Filename for the saved GIF")
    parser.add_argument("--gif_loop", type=int, default=3, help="Number of loops for GIF creation")

    args = parser.parse_args()
    main(args)

