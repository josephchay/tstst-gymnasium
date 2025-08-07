import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle
import time
import os
from symphony import HybridSymphony, SequentialReplayBuffer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global parameters
option = 8  # LunarLanderContinuous-v3
burst = False
tr_noise = True
explore_time = 3000  # Reduced exploration time
tr_between_ep_init = 15
tr_between_ep_const = False
tr_per_step = 2  # Reduced training per step
start_test = 50
limit_step = 1000
limit_eval = 1000
num_episodes = 10000
start_episode = 0

# World model parameters
world_model_train_freq = 100  # Train world model every N episodes
world_model_epochs = 50
dream_frequency = 5  # Generate dreams every N episodes

total_rewards, total_steps, test_rewards, Q_learning = [], [], [], False
hidden_dim = 256
max_action = 1.0
fade_factor = 7
stall_penalty = 0.07
capacity = "medium"  # Reduced capacity for efficiency

# Environment setup
env_configs = {
    -1: ('Pendulum-v1', {}),
    0: ('MountainCarContinuous-v0', {}),
    1: ('HalfCheetah-v4', {}),
    2: ('Walker2d-v4', {'tr_between_ep_init': 70}),
    3: ('Humanoid-v4', {'tr_between_ep_init': 200}),
    4: ('HumanoidStandup-v4', {'limit_step': 300, 'limit_eval': 300, 'tr_between_ep_init': 70}),
    5: ('Ant-v4', {'max_action': 0.7}),
    6: ('BipedalWalker-v3', {'tr_between_ep_init': 40, 'burst': True, 'tr_noise': False, 'limit_step': 1000}),
    7: ('BipedalWalkerHardcore-v3', {'burst': True, 'tr_noise': False, 'tr_between_ep_init': 0}),
    8: ('LunarLanderContinuous-v3', {'limit_step': 700, 'limit_eval': 700}),
    9: ('Pusher-v4', {'limit_step': 300, 'limit_eval': 200}),
    10: ('Swimmer-v4', {'burst': True})
}

env_name, env_params = env_configs[option]

# Apply environment-specific parameters
for key, value in env_params.items():
    if key in locals():
        locals()[key] = value

# Create environments
env = gym.make(env_name)
env_test = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print(f'Environment: {env_name}')
print(f'State dim: {state_dim}, Action dim: {action_dim}')
print(f'Action space high: {env.action_space.high}')

# Handle action scaling
if env.action_space.is_bounded():
    max_action = max_action * torch.FloatTensor(env.action_space.high).to(device)
else:
    max_action = max_action * 1.0

# Initialize components
replay_buffer = SequentialReplayBuffer(state_dim, action_dim, capacity, device)
algo = HybridSymphony(state_dim, action_dim, hidden_dim, device, max_action, burst, tr_noise)

def init_weights(m):
    """Initialize network weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def testing(env_test, limit_eval, test_episodes):
    """Evaluate the current policy"""
    if test_episodes < 1:
        return
    
    print(f"Validation... {test_episodes} episodes")
    episode_returns = []
    
    for test_episode in range(test_episodes):
        state = env_test.reset()[0]
        episode_reward = 0
        
        for steps in range(1, limit_eval + 1):
            action = algo.select_action(state, replay_buffer, mean=True)
            next_state, reward, done, _, _ = env_test.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_returns.append(episode_reward)
        avg_return = np.mean(episode_returns)
        
        if test_episode < 3 or test_episode % max(1, test_episodes // 5) == 0:
            print(f"Test {test_episode}: Return = {episode_reward:.2f}, "
                  f"Average = {avg_return:.2f}, Steps: {steps}")
    
    return np.mean(episode_returns)

def save_models_and_buffer():
    """Save models and replay buffer"""
    try:
        # Save models
        torch.save(algo.actor.state_dict(), 'hybrid_actor_model.pt')
        torch.save(algo.critic.state_dict(), 'hybrid_critic_model.pt')
        torch.save(algo.critic_target.state_dict(), 'hybrid_critic_target_model.pt')
        torch.save(algo.world_model.state_dict(), 'world_model.pt')
        
        # Save buffer and training state
        save_dict = {
            'buffer': replay_buffer,
            'x_coor': algo.actor.noise.x_coor,
            'total_rewards': total_rewards,
            'total_steps': total_steps,
            'Q_learning': Q_learning,
            'start_episode': len(total_rewards),
            'world_model_ready': algo.world_model_ready,
            'dream_ratio': algo.dream_ratio,
            'world_model_loss_history': algo.world_model_loss_history
        }
        
        with open('hybrid_replay_buffer.pkl', 'wb') as file:
            pickle.dump(save_dict, file)
            
        print("Models and buffer saved successfully")
    except Exception as e:
        print(f"Error saving models: {e}")

def load_models_and_buffer():
    """Load models and replay buffer"""
    global replay_buffer, total_rewards, total_steps, Q_learning, start_episode
    
    try:
        print("Loading buffer...")
        with open('hybrid_replay_buffer.pkl', 'rb') as file:
            save_dict = pickle.load(file)
        
        # Load noise state
        if 'x_coor' in save_dict:
            algo.actor.noise.x_coor = save_dict['x_coor']
        
        # Load buffer
        if 'buffer' in save_dict:
            loaded_buffer = save_dict['buffer']
            if hasattr(loaded_buffer, 'device'):
                replay_buffer = loaded_buffer
                replay_buffer.device = device
                # Ensure tensors are on correct device
                replay_buffer.states = replay_buffer.states.to(device)
                replay_buffer.actions = replay_buffer.actions.to(device)
                replay_buffer.rewards = replay_buffer.rewards.to(device)
                replay_buffer.next_states = replay_buffer.next_states.to(device)
                replay_buffer.dones = replay_buffer.dones.to(device)
                replay_buffer.episode_ids = replay_buffer.episode_ids.to(device)
        
        # Load training history
        total_rewards = save_dict.get('total_rewards', [])
        total_steps = save_dict.get('total_steps', [])
        Q_learning = save_dict.get('Q_learning', False)
        start_episode = save_dict.get('start_episode', 0)
        
        # Load world model state
        algo.world_model_ready = save_dict.get('world_model_ready', False)
        algo.dream_ratio = save_dict.get('dream_ratio', 0.0)
        algo.world_model_loss_history = save_dict.get('world_model_loss_history', [])
        
        print(f"Buffer loaded successfully. Length: {len(replay_buffer)}, Start episode: {start_episode}")
        
        if len(replay_buffer) >= explore_time and not Q_learning:
            Q_learning = True
            replay_buffer.find_min_max()
            print("Q-learning enabled")
            
    except Exception as e:
        print(f"Problem loading buffer: {e}")
        
    try:
        print("Loading models...")
        
        # Load models if they exist
        if os.path.exists('hybrid_actor_model.pt'):
            algo.actor.load_state_dict(torch.load('hybrid_actor_model.pt', 
                                                map_location=device, weights_only=True))
        if os.path.exists('hybrid_critic_model.pt'):
            algo.critic.load_state_dict(torch.load('hybrid_critic_model.pt', 
                                                 map_location=device, weights_only=True))
        if os.path.exists('hybrid_critic_target_model.pt'):
            algo.critic_target.load_state_dict(torch.load('hybrid_critic_target_model.pt', 
                                                        map_location=device, weights_only=True))
        if os.path.exists('world_model.pt'):
            algo.world_model.load_state_dict(torch.load('world_model.pt', 
                                                       map_location=device, weights_only=True))
            
        print('Models loaded successfully')
        
        # Run a quick test
        if Q_learning:
            test_score = testing(env_test, limit_eval, 3)
            print(f"Current performance: {test_score:.2f}")
            
    except Exception as e:
        print(f"Problem loading models: {e}")

# Load existing models and buffer
load_models_and_buffer()

# Training loop
print(f"Starting training from episode {start_episode}")

for episode in range(start_episode, num_episodes):
    rewards = []
    state = env.reset()[0]
    
    # Adaptive training between episodes
    rb_len = len(replay_buffer)
    tr_between_ep = tr_between_ep_init
    
    if not tr_between_ep_const and tr_between_ep_init >= 100 and rb_len >= 200000:
        tr_between_ep = rb_len // 3000
    elif not tr_between_ep_const and tr_between_ep_init < 100 and rb_len >= 3000 * tr_between_ep_init:
        tr_between_ep = rb_len // 3000
    
    # Training between episodes
    if Q_learning and rb_len >= 128:
        for _ in range(min(tr_between_ep, 100)):  # Cap training steps
            real_batch = replay_buffer.sample()
            
            # Generate dreams if ready
            dream_batch = None
            if (algo.world_model_ready and 
                episode % dream_frequency == 0):
                dream_batch = algo.generate_dreams(real_batch)
            
            algo.train(real_batch, dream_batch)
    
    # Start training after exploration
    if rb_len >= explore_time and not Q_learning:
        replay_buffer.find_min_max()
        print("Started Q-learning")
        Q_learning = True
        
        # Initial training boost
        if rb_len >= 128:
            for _ in range(64):
                algo.train(replay_buffer.sample())
    
    # Episode execution
    for episode_steps in range(1, limit_step + 1):
        # Select action
        if rb_len < explore_time:
            action = env.action_space.sample()  # Random exploration
        else:
            action = algo.select_action(state, replay_buffer)
        
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        
        # Environment-specific reward modifications
        modified_reward = reward
        if env_name.find("Ant") != -1:
            if next_state[1] < 0.4:
                done = True
            if next_state[1] > 1e-3:
                modified_reward += math.log(next_state[1])
        elif env_name.find("Humanoid-") != -1:
            modified_reward += next_state[0]
        elif env_name.find("LunarLander") != -1:
            if reward == -100.0:
                modified_reward = -50.0
        elif env_name.find("BipedalWalkerHardcore") != -1:
            if reward == -100.0:
                modified_reward = -25.0
        
        # Add to replay buffer
        replay_buffer.add(state, action, modified_reward + 1.0, next_state, done)
        
        # Training per step
        if Q_learning and rb_len >= 128:
            for _ in range(tr_per_step):
                real_batch = replay_buffer.sample()
                algo.train(real_batch)
        
        state = next_state
        if done:
            break
    
    # Record episode results
    total_rewards.append(np.sum(rewards))
    total_steps.append(episode_steps)
    average_reward = np.mean(total_rewards[-100:])
    
    print(f"Episode {episode}: Return = {total_rewards[-1]:.2f}, "
          f"Avg 100 = {average_reward:.2f}, Steps = {episode_steps}, "
          f"Buffer = {len(replay_buffer)}, Dream ratio = {algo.dream_ratio:.3f}")
    
    # Train world model periodically
    if (episode % world_model_train_freq == 0 and episode > 0 and 
        len(replay_buffer) >= 1000 and Q_learning):
        
        print(f"\nTraining world model at episode {episode}")
        losses = algo.train_world_model(replay_buffer, world_model_epochs)
        algo.update_curriculum()
        
        if losses:
            print(f"World model training complete. Final loss: {losses[-1]:.6f}\n")
    
    # Save models and buffer
    if Q_learning and episode % 20 == 0:
        save_models_and_buffer()
    
    # Validation testing
    if episode >= start_test and episode % 50 == 0:
        test_score = testing(env_test, limit_eval, 5)
        test_rewards.append(test_score)
        print(f"Test score: {test_score:.2f}\n")
    
    # Early stopping for exceptional performance
    if len(total_rewards) >= 100:
        recent_avg = np.mean(total_rewards[-50:])
        if env_name == "LunarLanderContinuous-v3" and recent_avg > 250:
            print(f"Early stopping - solved! Average reward: {recent_avg:.2f}")
            break
        elif env_name.find("Humanoid") != -1 and recent_avg > 6000:
            print(f"Early stopping - solved! Average reward: {recent_avg:.2f}")
            break

print("Training completed!")

# Final save and test
save_models_and_buffer()
final_score = testing(env_test, limit_eval, 10)
print(f"Final test score: {final_score:.2f}")

# Performance summary
if total_rewards:
    print(f"\nTraining Summary:")
    print(f"Total episodes: {len(total_rewards)}")
    print(f"Best episode: {max(total_rewards):.2f}")
    print(f"Final 100 episode average: {np.mean(total_rewards[-100:]):.2f}")
    print(f"World model ready: {algo.world_model_ready}")
    print(f"Final dream ratio: {algo.dream_ratio:.3f}")
