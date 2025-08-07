import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle
import time
import os
from symphony import UltraFastSymphony, PrioritizedReplayBuffer, get_adaptive_training_freq
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimized global parameters for maximum speed
option = 8  # LunarLanderContinuous-v3
burst = False
tr_noise = True
explore_time = 500          # Drastically reduced from 5000
tr_between_ep_init = 25     # Increased from 15
tr_between_ep_const = False
tr_per_step = 6             # Increased from 3
start_test = 25             # Earlier testing
limit_step = 2000
limit_eval = 2000
num_episodes = 10000
start_episode = 0

# Aggressive world model and dreaming parameters
world_model_train_freq = 25     # Much more frequent (was 100+)
world_model_epochs = 15         # Reduced for speed (was 50+)
dream_frequency = 2             # Very frequent dreaming
early_world_model_threshold = 150  # Start world model training very early

total_rewards, total_steps, test_rewards, Q_learning = [], [], [], False
hidden_dim = 256
max_action = 1.0
fade_factor = 7
stall_penalty = 0.07
capacity = "medium"  # Balanced capacity for speed

# Environment configurations with speed optimizations
env_configs = {
    -1: ('Pendulum-v1', {'explore_time': 200, 'tr_per_step': 4}),
    0: ('MountainCarContinuous-v0', {'explore_time': 300, 'tr_per_step': 5}),
    1: ('HalfCheetah-v4', {'tr_between_ep_init': 40, 'explore_time': 800}),
    2: ('Walker2d-v4', {'tr_between_ep_init': 50, 'explore_time': 600}),
    3: ('Humanoid-v4', {'tr_between_ep_init': 80, 'explore_time': 1000}),
    4: ('HumanoidStandup-v4', {'limit_step': 300, 'limit_eval': 300, 'tr_between_ep_init': 50, 'explore_time': 800}),
    5: ('Ant-v4', {'max_action': 0.7, 'tr_between_ep_init': 40, 'explore_time': 600}),
    6: ('BipedalWalker-v3', {'tr_between_ep_init': 30, 'burst': True, 'tr_noise': False, 'explore_time': 400}),
    7: ('BipedalWalkerHardcore-v3', {'burst': True, 'tr_noise': False, 'tr_between_ep_init': 20, 'explore_time': 600}),
    8: ('LunarLanderContinuous-v3', {'limit_step': 700, 'limit_eval': 700, 'explore_time': 500, 'tr_per_step': 6}),
    9: ('Pusher-v4', {'limit_step': 300, 'limit_eval': 200, 'explore_time': 400}),
    10: ('Swimmer-v4', {'burst': True, 'explore_time': 300})
}

env_name, env_params = env_configs[option]

# Apply environment-specific optimized parameters
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
print(f'Optimized explore_time: {explore_time}')

# Handle action scaling
if env.action_space.is_bounded():
    max_action = max_action * torch.FloatTensor(env.action_space.high).to(device)
else:
    max_action = max_action * 1.0

# Initialize optimized components
replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, capacity, device)
algo = UltraFastSymphony(state_dim, action_dim, hidden_dim, device, max_action, burst, tr_noise)

def init_weights(m):
    """Optimized weight initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)  # Better than Xavier for ReLU
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def testing(env_test, limit_eval, test_episodes):
    """Fast evaluation with reduced episodes"""
    if test_episodes < 1:
        return 0.0
    
    print(f"Fast validation... {test_episodes} episodes")
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
        
        if test_episode < 2 or test_episode == test_episodes - 1:
            print(f"Test {test_episode}: Return = {episode_reward:.2f}, Steps: {steps}")
    
    avg_return = np.mean(episode_returns)
    print(f"Average test return: {avg_return:.2f}")
    return avg_return

def save_models_and_buffer():
    """Optimized saving with error handling"""
    try:
        # Save models
        torch.save(algo.actor.state_dict(), 'ultra_actor_model.pt')
        torch.save(algo.critic.state_dict(), 'ultra_critic_model.pt')
        torch.save(algo.critic_target.state_dict(), 'ultra_critic_target_model.pt')
        torch.save(algo.world_model.state_dict(), 'ultra_world_model.pt')
        
        # Save training state
        save_dict = {
            'buffer': replay_buffer,
            'x_coor': algo.actor.noise.x_coor,
            'total_rewards': total_rewards,
            'total_steps': total_steps,
            'Q_learning': Q_learning,
            'start_episode': len(total_rewards),
            'world_model_ready': algo.world_model_ready,
            'dream_ratio': algo.dream_ratio,
            'world_model_losses': algo.world_model_losses
        }
        
        with open('ultra_replay_buffer.pkl', 'wb') as file:
            pickle.dump(save_dict, file)
            
        print("Ultra-fast models saved successfully")
    except Exception as e:
        print(f"Error saving models: {e}")

def load_models_and_buffer():
    """FIXED: Optimized loading with proper device handling"""
    global replay_buffer, total_rewards, total_steps, Q_learning, start_episode
    
    try:
        print("Loading buffer...")
        with open('ultra_replay_buffer.pkl', 'rb') as file:
            save_dict = pickle.load(file)
        
        # Load noise state
        if 'x_coor' in save_dict:
            algo.actor.noise.x_coor = save_dict['x_coor']
        
        # CRITICAL FIX: Load buffer with proper device migration
        if 'buffer' in save_dict:
            loaded_buffer = save_dict['buffer']
            if hasattr(loaded_buffer, 'device'):
                replay_buffer = loaded_buffer
                # FORCE all tensors to correct device
                replay_buffer.to_device(device)
        
        # Load training history
        total_rewards = save_dict.get('total_rewards', [])
        total_steps = save_dict.get('total_steps', [])
        Q_learning = save_dict.get('Q_learning', False)
        start_episode = save_dict.get('start_episode', 0)
        
        # Load world model state
        algo.world_model_ready = save_dict.get('world_model_ready', False)
        algo.dream_ratio = save_dict.get('dream_ratio', 0.0)
        algo.world_model_losses = save_dict.get('world_model_losses', [])
        
        print(f"Buffer loaded. Length: {len(replay_buffer)}, Start episode: {start_episode}")
        
        if len(replay_buffer) >= explore_time and not Q_learning:
            Q_learning = True
            replay_buffer.find_min_max()
            print("Q-learning enabled from saved state")
            
    except Exception as e:
        print(f"Problem loading buffer: {e}")
        
    try:
        print("Loading models...")
        
        # CRITICAL FIX: Load models with proper device mapping and error handling
        model_files = [
            ('ultra_actor_model.pt', algo.actor),
            ('ultra_critic_model.pt', algo.critic),
            ('ultra_critic_target_model.pt', algo.critic_target),
            ('ultra_world_model.pt', algo.world_model)
        ]
        
        for filename, model in model_files:
            if os.path.exists(filename):
                try:
                    # CRITICAL FIX: Force load to correct device and use weights_only=True
                    state_dict = torch.load(filename, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)
                    print(f"Loaded {filename}")
                except Exception as model_error:
                    print(f"Error loading {filename}: {model_error}")
        
        print('All available models loaded')
        
        # Quick performance test with error handling
        if Q_learning:
            try:
                test_score = testing(env_test, limit_eval, 2)
                print(f"Current performance: {test_score:.2f}")
            except Exception as test_error:
                print(f"Error during performance test: {test_error}")
            
    except Exception as e:
        print(f"Problem loading models: {e}")

# Load existing models and buffer
load_models_and_buffer()

# Ultra-fast training loop
print(f"Starting ultra-fast training from episode {start_episode}")

for episode in range(start_episode, num_episodes):
    episode_rewards = []  # Fixed: use episode_rewards instead of rewards to avoid conflict
    state = env.reset()[0]
    
    # Adaptive training parameters
    rb_len = len(replay_buffer)
    adaptive_tr_per_step = get_adaptive_training_freq(episode, rb_len)
    
    # Aggressive training between episodes
    tr_between_ep = tr_between_ep_init
    if not tr_between_ep_const and rb_len >= 3000 * tr_between_ep_init:
        tr_between_ep = min(rb_len // 2000, 100)  # More aggressive scaling
    
    # Intensive training between episodes
    if Q_learning and rb_len >= 128:
        for _ in range(min(tr_between_ep, 150)):  # Increased cap
            sample_result = replay_buffer.sample()
            if sample_result is None:
                break
                
            if len(sample_result) == 3:  # Prioritized buffer
                real_batch, weights, indices = sample_result
            else:
                real_batch = sample_result
                weights, indices = None, None
            
            # Generate dreams more frequently
            dream_batch = None
            if (algo.world_model_ready and episode % dream_frequency == 0):
                dream_batch = algo.generate_dreams(real_batch, n_dreams=30, dream_length=6)
            
            try:
                loss = algo.train(real_batch, dream_batch)
                
                # Update priorities if using prioritized replay
                if weights is not None and indices is not None:
                    # Compute TD errors for priority update
                    with torch.no_grad():
                        states, actions, rewards_tensor, next_states, dones = real_batch
                        current_qs = algo.critic(states, actions, united=False)
                        next_actions = algo.actor(next_states, mean=True)
                        next_q_target, _ = algo.critic_target(next_states, next_actions, united=True)
                        td_targets = rewards_tensor + (1 - dones) * 0.99 * next_q_target
                        td_errors = [abs(q - td_targets).mean().item() for q in current_qs[:3]]
                        avg_td_error = np.mean(td_errors)
                        replay_buffer.update_priorities(indices, [avg_td_error] * len(indices))
            except Exception as training_error:
                print(f"Training error: {training_error}")
    
    # Start training after short exploration
    if rb_len >= explore_time and not Q_learning:
        replay_buffer.find_min_max()
        print("Started ultra-fast Q-learning!")
        Q_learning = True
        
        # Intensive initial training
        if rb_len >= 128:
            print("Initial training boost...")
            for _ in range(100):  # Increased initial training
                sample_result = replay_buffer.sample()
                if sample_result is not None:
                    if len(sample_result) == 3:
                        batch, _, _ = sample_result
                    else:
                        batch = sample_result
                    try:
                        algo.train(batch)
                    except Exception as init_training_error:
                        print(f"Initial training error: {init_training_error}")
    
    # Episode execution with optimizations
    for episode_steps in range(1, limit_step + 1):
        # Optimized action selection with curiosity
        if rb_len < explore_time:
            if rb_len < 50:
                # Pure random exploration initially
                action = env.action_space.sample()
            else:
                # Mixed exploration with curiosity
                if np.random.random() < 0.4:  # 40% policy, 60% random
                    action = algo.select_action(state, replay_buffer)
                else:
                    action = env.action_space.sample()
        else:
            action = algo.select_action(state, replay_buffer)
        
        next_state, reward, done, info, _ = env.step(action)
        episode_rewards.append(reward)  # Fixed: use episode_rewards list
        
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
        
        # Add curiosity reward during exploration
        if rb_len >= 100 and rb_len < explore_time * 2:
            try:
                curiosity_reward = algo.curiosity.compute_intrinsic_reward(state, action, next_state)
                modified_reward += 0.5 * curiosity_reward
            except Exception as curiosity_error:
                pass  # Skip curiosity if there's an error
        
        # Compute TD error for prioritized replay
        td_error = 1.0
        if Q_learning and rb_len > 100:
            try:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_t = torch.FloatTensor(action).unsqueeze(0).to(device)
                    reward_t = torch.FloatTensor([modified_reward + 1.0]).unsqueeze(0).to(device)
                    next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                    done_t = torch.FloatTensor([done]).unsqueeze(0).to(device)
                    
                    if hasattr(replay_buffer, 'normalize'):
                        state_t = replay_buffer.normalize(state_t)
                        next_state_t = replay_buffer.normalize(next_state_t)
                    
                    current_q = algo.critic(state_t, action_t, united=True)[0]
                    next_action = algo.actor(next_state_t, mean=True)
                    next_q = algo.critic_target(next_state_t, next_action, united=True)[0]
                    target_q = reward_t + (1 - done_t) * 0.99 * next_q
                    td_error = abs(current_q - target_q).item()
            except Exception:
                td_error = 1.0
        
        # Add to prioritized buffer
        replay_buffer.add(state, action, modified_reward + 1.0, next_state, done, td_error)
        
        # Intensive training per step
        if Q_learning and rb_len >= 128:
            for _ in range(adaptive_tr_per_step):
                sample_result = replay_buffer.sample()
                if sample_result is None:
                    break
                    
                if len(sample_result) == 3:
                    batch, weights, indices = sample_result
                else:
                    batch = sample_result
                
                # More frequent dreaming during episode
                dream_batch = None
                if (algo.world_model_ready and episode_steps % 15 == 0):  # Every 15 steps
                    dream_batch = algo.generate_dreams(batch, n_dreams=20, dream_length=5)
                
                try:
                    algo.train(batch, dream_batch)
                except Exception as step_training_error:
                    pass  # Continue training even if one step fails
            
            # Train curiosity module frequently
            if episode_steps % 8 == 0:  # Every 8 steps
                sample_result = replay_buffer.sample()
                if sample_result is not None:
                    if len(sample_result) == 3:
                        curiosity_batch, _, _ = sample_result
                    else:
                        curiosity_batch = sample_result
                    try:
                        algo.curiosity.train_curiosity(curiosity_batch)
                    except Exception:
                        pass  # Continue if curiosity training fails
        
        state = next_state
        if done:
            break
    
    # Record episode results - Fixed: use episode_rewards
    total_rewards.append(np.sum(episode_rewards))
    total_steps.append(episode_steps)
    average_reward = np.mean(total_rewards[-100:])
    
    print(f"Ep {episode}: Rtrn = {total_rewards[-1]:.2f}, "
          f"Avg 100 = {average_reward:.2f}, Steps = {episode_steps}, "
          f"Buffer = {len(replay_buffer)}, Dream = {algo.dream_ratio:.3f}")
    
    # Aggressive world model training
    if (episode % world_model_train_freq == 0 and episode > 0 and 
        len(replay_buffer) >= early_world_model_threshold and Q_learning):
        
        print(f"\nTraining world model at episode {episode}")
        try:
            losses = algo.train_world_model(replay_buffer, world_model_epochs)
            
            if losses:
                print(f"World model training complete. Loss: {losses[-1]:.4f}")
                
                # Aggressive curriculum update
                if not algo.world_model_ready and np.mean(losses[-3:]) < 0.15:
                    algo.world_model_ready = True
                    algo.dream_ratio = 0.1
                    print("World model ready early due to low loss!")
        except Exception as world_model_error:
            print(f"World model training error: {world_model_error}")
    
    # Frequent saving
    if Q_learning and episode % 10 == 0:  # More frequent saves
        save_models_and_buffer()
    
    # Fast validation testing
    if episode >= start_test and episode % 25 == 0:  # More frequent testing
        try:
            test_score = testing(env_test, limit_eval, 3)  # Fewer test episodes
            test_rewards.append(test_score)
            print(f"Test score: {test_score:.2f}")
        except Exception as test_error:
            print(f"Testing error: {test_error}")
    
    # Aggressive early stopping
    if len(total_rewards) >= 50:  # Earlier early stopping check
        recent_avg = np.mean(total_rewards[-25:])  # Shorter window
        if env_name == "LunarLanderContinuous-v3" and recent_avg > 220:
            print(f"Early stopping - solved! Average: {recent_avg:.2f}")
            break
        elif env_name.find("Humanoid") != -1 and recent_avg > 5500:
            print(f"Early stopping - solved! Average: {recent_avg:.2f}")
            break
        elif env_name.find("Walker2d") != -1 and recent_avg > 3500:
            print(f"Early stopping - solved! Average: {recent_avg:.2f}")
            break
        elif env_name.find("HalfCheetah") != -1 and recent_avg > 8000:
            print(f"Early stopping - solved! Average: {recent_avg:.2f}")
            break

print("Ultra-fast training completed!")

# Final evaluation and summary
save_models_and_buffer()
try:
    final_score = testing(env_test, limit_eval, 5)
    print(f"Final test score: {final_score:.2f}")
except Exception as final_test_error:
    print(f"Final test error: {final_test_error}")

# Performance summary
if total_rewards:
    print(f"\nUltra-Fast Training Summary:")
    print(f"Total episodes: {len(total_rewards)}")
    print(f"Best episode: {max(total_rewards):.2f}")
    print(f"Final 50 episode average: {np.mean(total_rewards[-50:]):.2f}")
    print(f"World model ready: {algo.world_model_ready}")
    print(f"Final dream ratio: {algo.dream_ratio:.3f}")
    print(f"Exploration completed at episode: {min(explore_time//10, len(total_rewards))}")
    
    # Calculate learning efficiency
    if len(total_rewards) >= 100:
        early_avg = np.mean(total_rewards[:50])
        late_avg = np.mean(total_rewards[-50:])
        improvement = late_avg - early_avg
        print(f"Learning improvement: {improvement:.2f}")
        print(f"Learning rate: {improvement/len(total_rewards):.3f} per episode")
