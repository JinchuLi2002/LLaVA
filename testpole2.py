import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import gym
from collections import namedtuple, deque
from PIL import Image
from time import sleep
import os

# LLaVA imports (as in your original code)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

###########################
# Reward Function (VLM)
###########################
class RewardFunction:
    def __init__(self, model_path, model_base=None, conv_mode=None, load_8bit=False, load_4bit=False, device='cuda', save_samples=False, max_samples=10, samples_dir='samples'):
        disable_torch_init()
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            self.model_name,
            load_8bit,
            load_4bit,
            device=device
        )

        # Determine conversation mode
        if "llama-2" in self.model_name.lower():
            inferred_conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            inferred_conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            inferred_conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            inferred_conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            inferred_conv_mode = "mpt"
        else:
            inferred_conv_mode = "llava_v0"

        if conv_mode is not None and conv_mode != inferred_conv_mode:
            print("[WARNING] Manually specified conv_mode differs from inferred. Using specified:", conv_mode)
            self.conv_mode = conv_mode
        else:
            self.conv_mode = inferred_conv_mode

        self.conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles

        # Parameters for saving samples
        self.save_samples = save_samples
        self.max_samples = max_samples
        self.samples_dir = samples_dir
        self.sample_count = 0

        if self.save_samples:
            os.makedirs(self.samples_dir, exist_ok=True)
            print(f"Samples will be saved to '{self.samples_dir}/' directory.")

    def get_reward(self, image, episode, step):
        """
        Takes a PIL Image, processes it through the llava model,
        and returns a float reward in [0.0, 1.0].
        Additionally, saves the first `max_samples` images to `samples_dir`.
        """
        # Save sample images if within the limit
        if self.save_samples and self.sample_count < self.max_samples:
            sample_filename = f"episode_{episode}_step_{step}.png"
            sample_path = os.path.join(self.samples_dir, sample_filename)
            image.save(sample_path)
            self.sample_count += 1
            print(f"Saved sample image: {sample_path}")

        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # New prompt that instructs the model to return a number between 0.0 and 1.0
        prompt = (
            f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
            "You are a teacher evaluating how well the pole is balanced. "
            "Please look at the given image and respond with a single floating point number between 0.0 and 1.0, "
            "where 0.0 means the pole is completely off balance and 1.0 means the pole is perfectly balanced."
        )

        # Reset conversation
        self.conv = conv_templates[self.conv_mode].copy()
        self.conv.append_message(self.roles[0], prompt)
        self.conv.append_message(self.roles[1], None)
        full_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.model.device)

        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=False,
                max_new_tokens=100,
                streamer=streamer,
                use_cache=True
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        self.conv.messages[-1][-1] = output_text

        # Extract float from the output_text
        # We assume model will produce something like "0.85" or "0.5"
        try:
            txt = output_text.strip().split()[0]
            if txt[-4:] == "USER":
                txt = txt[:-4]
            reward = float(txt)
            reward = max(0.0, min(1.0, reward))  # Clamp between 0 and 1
        except:
            reward = 0.0
            print(f"[Warning] Failed to parse reward from model output: '{output_text}'. Assigning reward=0.0")

        return reward

###########################
# DQN Agent and Replay Buffer
###########################
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

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

class DQNAgent:
    def __init__(self, state_size, action_size, seed, lr=0.0025):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        self.memory = ReplayBuffer(action_size, buffer_size=int(1e5), batch_size=64, seed=seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    def act(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.action_size)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # DQN Target Calculation
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # DQN Expected Q values
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau)*target_param.data)

###########################
# Main Training Code
###########################
def main():
    # Configuration Parameters
    model_path = "liuhaotian/llava-v1.5-7b"  # Example model path; adjust as needed
    save_samples = True
    max_samples = 10
    samples_dir = "samples"

    # Initialize RewardFunction with image saving enabled
    reward_fn = RewardFunction(
        model_path=model_path,
        model_base=None,
        conv_mode=None,
        load_8bit=False,
        load_4bit=False,
        device=device,
        save_samples=save_samples,
        max_samples=max_samples,
        samples_dir=samples_dir
    )

    # Set up the CartPole environment with render_mode='rgb_array'
    env = gym.make("CartPole-v1", render_mode='rgb_array')

    # Training parameters
    num_episodes = 250
    max_steps_per_episode = 50  # Reduced to limit calls to the VLM
    epsilon_start = 1.0
    epsilon_end = 0.2
    epsilon_decay_rate = 0.99
    gamma = 0.99  # Increased gamma for better credit assignment
    lr = 0.0025
    update_frequency = 10
    seed = 170715

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Initialize the DQNAgent
    agent = DQNAgent(state_size=input_dim, action_size=output_dim, seed=seed, lr=lr)

    # Create directory for saving sample images if not already created
    if save_samples:
        os.makedirs(samples_dir, exist_ok=True)

    print("Starting Training...\n")
    # Training loop
    for episode in range(1, num_episodes + 1):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))

        for step in range(1, max_steps_per_episode + 1):
            action = agent.act(state, epsilon)
            step_result = env.step(action)

            # Extract step result
            if len(step_result) == 5:
                next_state, _, done, truncated, info = step_result
                done = done or truncated
            elif len(step_result) == 4:
                next_state, _, done, info = step_result
            else:
                raise ValueError("Unexpected step result format from env.step().")

            # Get the rendered image and compute reward using VLM
            frame = env.render()
            pil_image = Image.fromarray(frame).convert('RGB')
            vlm_reward = reward_fn.get_reward(pil_image, episode, step)

            # Step the agent
            agent.step(state, action, vlm_reward, next_state, done)
            state = next_state

            if done:
                break

        if episode % update_frequency == 0:
            print(f"Episode {episode}/{num_episodes} completed. Epsilon: {epsilon:.4f}")

    print("\nTraining Completed!")

    # Evaluate the trained agent
    test_episodes = 10
    episode_rewards = []

    print("\nStarting Evaluation...\n")
    for episode in range(1, test_episodes + 1):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_reward = 0
        done = False

        for step in range(1, max_steps_per_episode + 1):
            action = agent.act(state, eps=0.)  # No exploration during testing
            step_result = env.step(action)

            if len(step_result) == 5:
                next_state, _, done, truncated, info = step_result
                done = done or truncated
            elif len(step_result) == 4:
                next_state, _, done, info = step_result
            else:
                raise ValueError("Unexpected step result format from env.step().")

            frame = env.render()
            pil_image = Image.fromarray(frame).convert('RGB')
            vlm_reward = reward_fn.get_reward(pil_image, episode, step)
            episode_reward += vlm_reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Test Episode {episode}: Reward = {episode_reward:.2f}")

    average_reward = np.mean(episode_rewards)
    print(f"\nAverage VLM-based reward over {test_episodes} test episodes: {average_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
