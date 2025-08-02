

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import cv2
from collections import deque
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import imageio
from tqdm import tqdm
import json
import gymnasium as gym

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CarRacingWrapper(gym.Wrapper):

    def __init__(self, env, stack_size=4, img_size=84, skip_frames=2):
        super(CarRacingWrapper, self).__init__(env)
        self.stack_size = stack_size
        self.img_size = img_size
        self.skip_frames = skip_frames
        self.frames = deque(maxlen=stack_size)
        
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(stack_size, img_size, img_size), 
            dtype=np.float32
        )
        
        
        self.episode_reward = 0
        self.episode_length = 0
        self.consecutive_negative_reward = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.consecutive_negative_reward = 0
        
        processed_obs = self.preprocess_frame(obs)
        
        for _ in range(self.stack_size):
            self.frames.append(processed_obs)
        
        return np.array(self.frames, dtype=np.float32), info
    
    def step(self, action):
       
        action = np.array(action, dtype=np.float32)
        
        total_reward = 0
        
       
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
     
        processed_obs = self.preprocess_frame(obs)
        self.frames.append(processed_obs)
        
      
        shaped_reward = self.shape_reward(total_reward, obs)
        
        
        if shaped_reward < -0.1:
            self.consecutive_negative_reward += 1
        else:
            self.consecutive_negative_reward = 0
        
        
        done = terminated or truncated
        if self.consecutive_negative_reward > 50:  
            done = True
            shaped_reward -= 10  
        
        self.episode_reward += shaped_reward
        self.episode_length += 1
        
        return np.array(self.frames, dtype=np.float32), shaped_reward, done, truncated, info
    
    def preprocess_frame(self, frame):
        """Improved frame preprocessing"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
        cropped = gray[0:84, 6:90]  
        
      
        resized = cv2.resize(cropped, (self.img_size, self.img_size), 
                           interpolation=cv2.INTER_AREA)
        
        
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def shape_reward(self, reward, obs):
        
        if reward > 0:  
            reward *= 1.5  
        elif reward <= -0.1:  
            reward *= 1.5  
        
        return reward

class PPONetwork(nn.Module):
   
    def __init__(self, input_shape, action_dim, hidden_dim=512):
        super(PPONetwork, self).__init__()
        
       
        self.conv_layers = nn.Sequential(
           
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
           
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
      
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
          
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.shape[1]
        
      
        self.shared_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
       
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  
        )
        
  
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
       
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
      
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        conv_features = self.conv_layers(x)
        shared_features = self.shared_layers(conv_features)
        
      
        action_mean = self.actor_mean(shared_features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std.clamp(-20, 2))  
        
    
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, x, action=None):
        action_mean, action_std, value = self.forward(x)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)

class PPOAgent:
    
    def __init__(self, input_shape, action_dim, lr=2.5e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, value_coef=0.5, 
                 entropy_coef=0.01, max_grad_norm=0.5, device=None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
       
        self.network = PPONetwork(input_shape, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)
       
        self.reset_storage()
        
   
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
    
    def reset_storage(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action_mean, _, value = self.network(state_tensor)
                action = action_mean
                log_prob = torch.zeros(1)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        
        returns = np.array(returns)
        advantages = returns - values[:-1]
        
        return returns, advantages
    
    def update(self, next_value, epochs=4, batch_size=64):
        if len(self.states) < batch_size:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'total_loss': 0}
        
       
        returns, advantages = self.compute_gae(next_value)
        
      
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
       
        for epoch in range(epochs):
         
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
               
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
         
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
              
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
               
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_ratio, self.clip_ratio
                )
                value_loss_1 = F.mse_loss(values, batch_returns)
                value_loss_2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_loss_1, value_loss_2)
                
              
                entropy_loss = -entropy.mean()
                
             
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
             
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
    
        self.scheduler.step()
        
        
        self.reset_storage()
        
     
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_value_loss = np.mean(value_losses) if value_losses else 0
        avg_entropy_loss = np.mean(entropy_losses) if entropy_losses else 0
        avg_total_loss = avg_policy_loss + self.value_coef * avg_value_loss + self.entropy_coef * avg_entropy_loss
        
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        self.training_stats['entropy_losses'].append(avg_entropy_loss)
        self.training_stats['total_losses'].append(avg_total_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss
        }
    
    def save(self, filepath):
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
       
    
    def load(self, filepath):
        try:
            
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying alternative loading method...")
            try:
                
                import torch.serialization
                with torch.serialization.safe_globals(['numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar']):
                    checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except:
               
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        print(f"Model loaded from {filepath}")

def train_ppo_agent(episodes=100000, timesteps_per_episode=1000, update_frequency=2048, 
                   save_frequency=None, eval_frequency=2000):
   
    
    
    env = gym.make('CarRacing-v3', render_mode=None)
    env = CarRacingWrapper(env)
    
    
    input_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(input_shape, action_dim, lr=2.5e-4)
    
   
    episode_rewards = []
    episode_lengths = []
    timestep = 0
    best_avg_reward = -float('inf')
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Environment: CarRacing-v3")
    print(f"Observation shape: {input_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {agent.device}")
    print("=" * 50)
    
  
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(timesteps_per_episode):
            
            action, log_prob, value = agent.get_action(state)
            
          
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
           
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            
            if timestep % update_frequency == 0:
                next_value = 0 if done else agent.get_action(state)[2]
                losses = agent.update(next_value)
                
                if losses['total_loss'] > 0: 
                    print(f"Timestep {timestep:6d} | "
                          f"Policy Loss: {losses['policy_loss']:.4f} | "
                          f"Value Loss: {losses['value_loss']:.4f} | "
                          f"Entropy Loss: {losses['entropy_loss']:.4f}")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent.training_stats['episode_rewards'].append(episode_reward)
        agent.training_stats['episode_lengths'].append(episode_length)
        
     
        avg_reward = np.mean(episode_rewards[-100:])
        
        print(f"Episode {episode + 1:4d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Length: {episode_length:4d} | "
              f"Avg(100): {avg_reward:7.2f}")
        
     
        if avg_reward > best_avg_reward and (episode + 1) % 100 == 0:
            best_avg_reward = avg_reward
            agent.save('ppo_carracing_best.pth')
        
       
        if (episode + 1) % eval_frequency == 0:
            print("Running evaluation...")
            eval_reward = evaluate_agent(agent, episodes=5, render=False)
            print(f"Evaluation after episode {episode + 1}: {eval_reward:.2f}")
        
       
        if avg_reward >= 900:
            print(f"Environment solved in {episode + 1} episodes!")
            agent.save('ppo_carracing_solved.pth')
            break
    
    
    agent.save('ppo_carracing_final.pth')
    
    
    plot_training_progress(episode_rewards, agent.training_stats)
    
    return agent

def evaluate_agent(agent, episodes=10, render=False):
   
    render_mode = 'human' if render else None
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    env = CarRacingWrapper(env)
    
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 2000
        
        while not done and steps < max_steps:
           
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        if render:
            print(f"Evaluation Episode {episode + 1}: {episode_reward:.2f}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    if not render:
        return avg_reward
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    print(f"Episodes >= 900: {sum(1 for r in total_rewards if r >= 900)}/{len(total_rewards)}")
    
    return total_rewards

def create_demo_video(agent, video_path='demo_video.mp4', episodes=3):
    try:
        
        env = gym.make('CarRacing-v3', render_mode='rgb_array')
        env = CarRacingWrapper(env)
        
        frames = []
        
        print(f"Creating demo video with {episodes} episodes...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            episode_frames = 0
            max_episode_frames = 2000
            
            print(f"Recording episode {episode + 1}...")
            
            while not done and episode_frames < max_episode_frames:
               
                action, _, _ = agent.get_action(state, deterministic=True)
                state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                episode_reward += reward
                
               
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    episode_frames += 1
                except Exception as e:
                    print(f"Error rendering frame: {e}")
                    break
            
            print(f"Episode {episode + 1} reward: {episode_reward:.2f}, frames: {episode_frames}")
        
        env.close()
        
        if len(frames) == 0:
            print("No frames captured. Cannot create video.")
            return
        
       
        print(f"Saving video to {video_path}...")
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Video saved successfully! Total frames: {len(frames)}")
        except Exception as e:
            print(f"Error saving video: {e}")
            
            try:
                video_path_gif = video_path.replace('.mp4', '.gif')
                imageio.mimsave(video_path_gif, frames[::2], fps=15) 
                print(f"Saved as GIF instead: {video_path_gif}")
            except Exception as e2:
                print(f"Failed to save as GIF too: {e2}")
        
    except Exception as e:
        print(f"Error creating demo video: {e}")

def plot_training_progress(episode_rewards, training_stats):
    """Plot training progress with improved visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    if len(episode_rewards) >= 10:
        moving_avg = [np.mean(episode_rewards[max(0, i-9):i+1]) for i in range(len(episode_rewards))]
        axes[0, 0].plot(moving_avg, label='10-Episode Average', color='red', linewidth=2)
    if len(episode_rewards) >= 100:
        moving_avg_100 = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
        axes[0, 0].plot(moving_avg_100, label='100-Episode Average', color='green', linewidth=2)
    
    axes[0, 0].axhline(y=900, color='orange', linestyle='--', label='Target (900)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
  
    if training_stats['total_losses']:
        axes[0, 1].plot(training_stats['total_losses'], label='Total Loss', alpha=0.7)
        axes[0, 1].plot(training_stats['policy_losses'], label='Policy Loss', alpha=0.7)
        axes[0, 1].plot(training_stats['value_losses'], label='Value Loss', alpha=0.7)
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
   
    axes[1, 0].hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1, 0].axvline(900, color='orange', linestyle='--', label='Target: 900')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
  
    if training_stats['episode_lengths']:
        axes[1, 1].plot(training_stats['episode_lengths'], alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Episode Length')
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'training_progress_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved as {filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Improved PPO CarRacing Agent')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'demo'], 
                       default='train', help='Mode: train, eval, or demo')
    parser.add_argument('--model_path', type=str, default='ppo_carracing_best.pth',
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=100000,
                       help='Number of episodes for training')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training...")
       
        try:
            agent = train_ppo_agent(episodes=args.episodes)
            print("Training completed successfully!")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Models have been saved.")
        
    elif args.mode == 'eval':
        print("Loading trained agent...")
        env = gym.make('CarRacing-v3', render_mode=None)
        env = CarRacingWrapper(env)
        
        input_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        agent = PPOAgent(input_shape, action_dim)
        
        try:
            agent.load(args.model_path)
            print("Evaluating agent...")
            rewards = evaluate_agent(agent, episodes=args.eval_episodes, render=args.render)
            if isinstance(rewards, list):
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                episodes_above_900 = sum(1 for r in rewards if r >= 900)
                
                print(f"\n{'='*50}")
                print(f"EVALUATION RESULTS:")
                print(f"{'='*50}")
                print(f"Episodes: {len(rewards)}")
                print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
                print(f"Max Reward: {max(rewards):.2f}")
                print(f"Min Reward: {min(rewards):.2f}")
                print(f"Episodes >= 900: {episodes_above_900}/{len(rewards)} ({episodes_above_900/len(rewards)*100:.1f}%)")
                
                
        except FileNotFoundError:
            print(f" Model file {args.model_path} not found.")
            print("Please train the agent first using: python script.py --mode train")
        except Exception as e:
            print(f" Error loading model: {e}")
        
        env.close()
        
    elif args.mode == 'demo':
        print("Loading trained agent for demo...")
        env = gym.make('CarRacing-v3', render_mode=None)
        env = CarRacingWrapper(env)
        
        input_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        agent = PPOAgent(input_shape, action_dim)
        
        try:
            agent.load(args.model_path)
            print("Creating demo video...")
            create_demo_video(agent, episodes=3)
            print("Demo video creation completed!")
        except FileNotFoundError:
            print(f" Model file {args.model_path} not found.")
            print("Please train the agent first using: python script.py --mode train")
        except Exception as e:
            print(f" Error creating demo: {e}")
        
        env.close()

if __name__ == "__main__":
    main()