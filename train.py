import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle
import time
from symphony import Symphony, ReplayBuffer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#global parameters
option = 8
burst = False
tr_noise = True
explore_time = 5000
tr_between_ep_init = 15
tr_between_ep_const = False
tr_per_step = 3
start_test = 50
limit_step = 2000
limit_eval = 2000
num_episodes = 10000000
start_episode = 0
total_rewards, total_steps, test_rewards, Q_learning = [], [], [], False
hidden_dim = 256
max_action = 1.0
fade_factor = 7
stall_penalty = 0.07
capacity = "full"

# Enhanced training parameters
use_prioritized_sampling = False  # Can be enabled for harder environments
adaptive_training_schedule = True  # Adjust training frequency based on performance

if option == -1:
    env = gym.make('Pendulum-v1')
    env_test = gym.make('Pendulum-v1', render_mode="human")
elif option == 0:
    env = gym.make('MountainCarContinuous-v0')
    env_test = gym.make('MountainCarContinuous-v0', render_mode="human")
elif option == 1:
    env = gym.make('HalfCheetah-v4')
    env_test = gym.make('HalfCheetah-v4', render_mode="human")
elif option == 2:
    tr_between_ep_init = 70
    env = gym.make('Walker2d-v4')
    env_test = gym.make('Walker2d-v4', render_mode="human")
elif option == 3:
    tr_between_ep_init = 200
    env = gym.make('Humanoid-v4')
    env_test = gym.make('Humanoid-v4', render_mode="human")
elif option == 4:
    limit_step = 300
    limit_eval = 300
    tr_between_ep_init = 70
    env = gym.make('HumanoidStandup-v4')
    env_test = gym.make('HumanoidStandup-v4', render_mode="human")
elif option == 5:
    env = gym.make('Ant-v4')
    env_test = gym.make('Ant-v4', render_mode="human")
    angle_limit = 0.4
    max_action = 0.7
elif option == 6:
    tr_between_ep_init = 40
    burst = True
    tr_noise = False
    limit_step = int(1e+6)
    env = gym.make('BipedalWalker-v3')
    env_test = gym.make('BipedalWalker-v3', render_mode="human")
elif option == 7:
    burst = True
    tr_noise = False
    tr_between_ep_init = 0
    env = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
    env_test = gym.make('BipedalWalkerHardcore-v3')
elif option == 8:
    limit_step = 700
    limit_eval = 700
    env = gym.make('LunarLanderContinuous-v3')
    env_test = gym.make('LunarLanderContinuous-v3')
elif option == 9:
    limit_step = 300
    limit_eval = 200
    env = gym.make('Pusher-v4')
    env_test = gym.make('Pusher-v4', render_mode="human")
elif option == 10:
    burst = True
    env = gym.make('Swimmer-v4')
    env_test = gym.make('Swimmer-v4', render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('action space high', env.action_space.high)

max_action = max_action*torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else max_action*1.0

replay_buffer = ReplayBuffer(state_dim, action_dim, capacity, device, fade_factor, stall_penalty)
algo = Symphony(state_dim, action_dim, hidden_dim, device, max_action, burst, tr_noise)

def init_weights(m):
    if isinstance(m, nn.Linear): 
        torch.nn.init.xavier_uniform_(m.weight)

def testing(env, limit_step, test_episodes):
    if test_episodes < 1: 
        return
    print("Validation... ", test_episodes, " episodes")
    episode_return = []
    
    for test_episode in range(test_episodes):
        state = env.reset()[0]
        rewards = []
        
        for steps in range(1, limit_step+1):
            action = algo.select_action(state, replay_buffer, mean=True)
            next_state, reward, done, info, _ = env.step(action)
            rewards.append(reward)
            state = next_state
            if done: 
                break
        
        episode_return.append(np.sum(rewards))
        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode}: Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}")

# Loading existing models and replay buffer
try:
    print("loading buffer...")
    with open('replay_buffer', 'rb') as file:
        dict = pickle.load(file)
        algo.actor.noise.x_coor = dict['x_coor']
        
        # Handle both old and new buffer formats
        old_buffer = dict['buffer']
        if hasattr(old_buffer, 'priorities'):
            replay_buffer = old_buffer  # New format
        else:
            # Convert old format to new format
            replay_buffer = ReplayBuffer(state_dim, action_dim, capacity, device, fade_factor, stall_penalty)
            if hasattr(old_buffer, 'states') and len(old_buffer) > 0:
                copy_length = min(len(old_buffer), replay_buffer.capacity)
                replay_buffer.states[:copy_length] = old_buffer.states[:copy_length]
                replay_buffer.actions[:copy_length] = old_buffer.actions[:copy_length]
                replay_buffer.rewards[:copy_length] = old_buffer.rewards[:copy_length]
                replay_buffer.next_states[:copy_length] = old_buffer.next_states[:copy_length]
                replay_buffer.dones[:copy_length] = old_buffer.dones[:copy_length]
                replay_buffer.length = copy_length
                replay_buffer.step = copy_length
        
        total_rewards = dict['total_rewards']
        total_steps = dict['total_steps']
        average_steps = dict['average_steps']
        
        if len(replay_buffer) >= explore_time and not Q_learning: 
            Q_learning = True
        print('buffer loaded, buffer length', len(replay_buffer))
        start_episode = len(total_steps)
except:
    print("problem during loading buffer")

try:
    print("loading models...")
    algo.actor.load_state_dict(torch.load('actor_model.pt', weights_only=True))
    algo.critic.load_state_dict(torch.load('critic_model.pt', weights_only=True))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt', weights_only=True))
    print('models loaded')
    testing(env_test, limit_eval, 10)
except:
    print("problem during loading models")

# Enhanced performance tracking
recent_rewards = []
performance_trend = 0
stagnation_counter = 0

# Training loop
for i in range(start_episode, num_episodes):
    rewards = []
    state = env.reset()[0]
    
    # Enhanced adaptive training between episodes
    rb_len = len(replay_buffer)
    rb_len_threshold = 5000*tr_between_ep_init
    tr_between_ep = tr_between_ep_init
    
    if adaptive_training_schedule and Q_learning:
        # Adjust training frequency based on recent performance
        if len(recent_rewards) >= 10:
            current_avg = np.mean(recent_rewards[-10:])
            older_avg = np.mean(recent_rewards[-20:-10]) if len(recent_rewards) >= 20 else current_avg
            performance_trend = current_avg - older_avg
            
            if performance_trend > 0:  # Improving
                tr_between_ep = max(tr_between_ep_init // 2, tr_between_ep - 5)
            elif performance_trend < -50:  # Declining significantly
                tr_between_ep = min(tr_between_ep_init * 2, tr_between_ep + 10)
    
    if not tr_between_ep_const and tr_between_ep_init >= 100 and rb_len >= 350000: 
        tr_between_ep = rb_len//5000
    if not tr_between_ep_const and tr_between_ep_init < 100 and rb_len >= rb_len_threshold: 
        tr_between_ep = rb_len//5000

    if Q_learning: 
        time.sleep(0.5)
    
    # Decrease dependence on random seed
    if not Q_learning and rb_len == 1000:
        _ = [algo.actor.apply(init_weights) for x in range(7)]
        print("Actor reinitialized to decrease dependence on random seed")
    
    # Training between episodes with enhanced sampling
    if Q_learning and rb_len > 128:  # Safety check
        for _ in range(tr_between_ep):
            if use_prioritized_sampling and hasattr(replay_buffer, 'sample'):
                batch = replay_buffer.sample(prioritized=True)
            else:
                batch = replay_buffer.sample()
            algo.train(batch, replay_buffer)
    
    # Episode execution
    for episode_steps in range(1, limit_step+1):
        # Start training after exploration phase
        if rb_len >= explore_time and not Q_learning:
            replay_buffer.find_min_max()
            print("started training")
            Q_learning = True
            
            # Safe initial training
            if rb_len >= 128:
                _ = [algo.train(replay_buffer.sample(uniform=True), replay_buffer) for x in range(64)]
                _ = [algo.train(replay_buffer.sample(), replay_buffer) for x in range(64)]
        
        action = algo.select_action(state, replay_buffer)
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        
        # Environment-specific reward modifications
        if env.spec.id.find("Ant") != -1:
            if (next_state[1] < 0.4): done = True
            if next_state[1] > 1e-3: reward += math.log(next_state[1])
        elif env.spec.id.find("Humanoid-") != -1:
            reward += next_state[0]
        elif env.spec.id.find("LunarLander") != -1:
            if reward == -100.0: reward = -50.0
        elif env.spec.id.find("BipedalWalkerHardcore") != -1:
            if reward == -100.0: reward = -25.0
        
        replay_buffer.add(state, action, reward+1.0, next_state, done)
        
        # Training per step with safety checks
        if Q_learning and rb_len >= 128:
            for _ in range(tr_per_step):
                batch = replay_buffer.sample()
                algo.train(batch, replay_buffer)
        
        state = next_state
        if done: 
            break
    
    total_rewards.append(np.sum(rewards))
    recent_rewards.append(np.sum(rewards))
    if len(recent_rewards) > 100:
        recent_rewards.pop(0)
        
    average_reward = np.mean(total_rewards[-100:])
    total_steps.append(episode_steps)
    average_steps = np.mean(total_steps[-100:])
    
    # Enhanced progress reporting
    trend_info = ""
    if len(recent_rewards) >= 20:
        trend_info = f"| Trend: {performance_trend:+.1f}"
    
    print(f"Ep {i}: Rtrn = {total_rewards[-1]:.2f} | ep steps = {episode_steps} {trend_info}")
    
    if Q_learning:
        # Saving models and buffer
        if (i % 5 == 0):
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            
            with open('replay_buffer', 'wb') as file:
                pickle.dump({
                    'buffer': replay_buffer, 
                    'x_coor': algo.actor.noise.x_coor, 
                    'total_rewards': total_rewards, 
                    'total_steps': total_steps, 
                    'average_steps': average_steps
                }, file)
        
        # Validation testing
        if (i >= start_test and i % 50 == 0): 
            testing(env_test, limit_step=limit_eval, test_episodes=10)
