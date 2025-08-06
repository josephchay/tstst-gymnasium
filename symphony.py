#This is the enhanced Symphony algorithm with conservative improvements.
#Focuses on proven techniques: better sampling, improved noise schedules, and stability enhancements.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
import random

# random seeds
r1, r2, r3 = 830143436, 167430301, 2193498338
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)

#Rectified Hubber Error Loss Function
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function
def ReHaE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)

class ReSine(nn.Module):
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)

class FourierSeries(nn.Module):
    def __init__(self, hidden_dim, f_out):
        super().__init__()
        self.fft = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ReSine(),
            nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x):
        return self.fft(x)

# Enhanced Actor with improved exploration
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0, burst=False, tr_noise=True):
        super(Actor, self).__init__()
        self.input = nn.Linear(state_dim, hidden_dim)
        self.net = nn.Sequential(
            FourierSeries(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = torch.mean(max_action).item()
        self.noise = EnhancedNoise(max_action, burst, tr_noise)
        
    def forward(self, state, mean=False):
        x = self.input(state)
        x = self.max_action * self.net(x)
        if mean: return x
        x += self.noise.generate(x)
        return x.clamp(-self.max_action, self.max_action)

# Enhanced Critic with better stability
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()
        self.input = nn.Linear(state_dim + action_dim, hidden_dim)
        qA = FourierSeries(hidden_dim, 1)
        qB = FourierSeries(hidden_dim, 1)
        qC = FourierSeries(hidden_dim, 1)
        s2 = FourierSeries(hidden_dim, 1)
        self.nets = nn.ModuleList([qA, qB, qC, s2])
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        x = self.layer_norm(self.input(x))
        xs = [net(x) for net in self.nets]
        if not united: return xs
        qmin = torch.min(torch.stack(xs[:3], dim=-1), dim=-1).values
        return qmin, xs[3]

# Enhanced Replay Buffer with better sampling strategies
class EnhancedReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        capacity_dict = {"short": 100000, "medium": 300000, "full": 500000}
        self.capacity, self.length, self.device = capacity_dict[capacity], 0, device
        self.batch_size = min(max(128, self.length//500), 1024)
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs, self.step = [], np.array([]), np.array([]), 0
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty
        
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        
        # Add importance weights for better sampling
        self.priorities = torch.ones((self.capacity,), dtype=torch.float32).to(device)
        
        self.raw = True

    def find_min_max(self):
        if self.length == 0:
            return
        actual_length = min(self.length, self.capacity)
        self.min_values = torch.min(self.states[:actual_length], dim=0).values
        self.max_values = torch.max(self.states[:actual_length], dim=0).values
        self.min_values[torch.isinf(self.min_values)] = -1e+3
        self.max_values[torch.isinf(self.max_values)] = 1e+3
        self.min_values = 2.0*(torch.floor(10.0*self.min_values)/10.0).reshape(1, -1).to(self.device)
        self.max_values = 2.0*(torch.ceil(10.0*self.max_values)/10.0).reshape(1, -1).to(self.device)
        self.raw = False

    def normalize(self, state):
        if self.raw: return state
        state = torch.clamp(state, -1e+3, 1e+3)
        range_vals = self.max_values - self.min_values
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero
        state = 4.0 * (state - self.min_values) / range_vals - 2.0
        state[torch.isnan(state)] = 0.0
        return state

    def add(self, state, action, reward, next_state, done):
        if self.length < self.capacity:
            self.length += 1
        
        index = self.step % self.capacity
        self.states[index] = torch.FloatTensor(state).to(self.device)
        self.actions[index] = torch.FloatTensor(action).to(self.device)
        self.rewards[index] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[index] = torch.FloatTensor(next_state).to(self.device)
        self.dones[index] = torch.FloatTensor([done]).to(self.device)
        
        # Initialize priority for new transitions
        self.priorities[index] = self.priorities[:self.length].max() if self.length > 1 else 1.0
        
        self.step += 1
        if self.length > 100: 
            self.batch_size = min(max(128, self.length//500), 1024)

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        if len(indices) > 0:
            self.priorities[indices] = torch.FloatTensor(priorities).to(self.device) + 1e-6

    def generate_probs(self, uniform=False, prioritized=False):
        if self.length <= 0:
            self.indexes = np.array([])
            return None
            
        if uniform or self.length <= 100: 
            self.indexes = np.arange(self.length)
            return None

        if self.length >= self.capacity and hasattr(self, 'probs') and not prioritized: 
            return self.probs

        def fade(norm_index): 
            return np.tanh(self.fade_factor*norm_index**2)
        
        self.indexes = np.arange(self.length)
        
        if prioritized:
            # Use priority-based sampling
            priorities = self.priorities[:self.length].cpu().numpy()
            weights = priorities ** 0.6  # Priority exponent
        else:
            # Use temporal fading
            weights = 1e-7 * fade(self.indexes/self.length)
        
        self.probs = weights / np.sum(weights)
        return self.probs

    def sample(self, uniform=False, prioritized=False):
        if self.length == 0:
            # Return empty tensors with correct dimensions
            return (
                torch.zeros((0, self.states.shape[1])).to(self.device),
                torch.zeros((0, self.actions.shape[1])).to(self.device),
                torch.zeros((0, 1)).to(self.device),
                torch.zeros((0, self.states.shape[1])).to(self.device),
                torch.zeros((0, 1)).to(self.device)
            )
        
        actual_batch_size = min(self.batch_size, self.length)
        probs = self.generate_probs(uniform, prioritized)
        
        indices = self.random.choice(self.indexes, p=probs, size=actual_batch_size)
        
        return (
            self.normalize(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self.normalize(self.next_states[indices]),
            self.dones[indices]
        )

    def __len__(self):
        return self.length

# Enhanced Symphony algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, burst=False, tr_noise=True):
        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action, burst, tr_noise).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Enhanced optimizers with gradient clipping
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4, weight_decay=1e-5)
        
        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0
        self.tr_step = 0
        
        # Enhanced training parameters
        self.grad_clip_value = 1.0
        self.target_update_freq = 2

    def select_action(self, state, replay_buffer=None, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1, self.state_dim).to(self.device)
            if replay_buffer: state = replay_buffer.normalize(state)
            action = self.actor(state, mean=mean)
            return action.cpu().data.numpy().flatten()

    def train(self, batch, replay_buffer=None):
        self.tr_step += 1
        state, action, reward, next_state, done = batch
        
        # Skip training if batch is empty
        if state.size(0) == 0:
            return torch.tensor(0.0)
        
        critic_loss = self.critic_update(state, action, reward, next_state, done)
        actor_loss = self.actor_update(state, next_state)
        
        return actor_loss

    def critic_update(self, state, action, reward, next_state, done):
        # Enhanced target update with less frequent updates
        if self.tr_step % self.target_update_freq == 0:
            with torch.no_grad():
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(0.995*target_param.data + 0.005*param)
        
        with torch.no_grad():
            next_action = self.actor(next_state, mean=True)
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward + (1-done) * 0.99 * q_next_target
            
            # Enhanced variance estimation
            reward_var = torch.var(reward) if reward.numel() > 1 else torch.tensor(0.0).to(self.device)
            s2_value = 3e-3 * (3e-3 * reward_var + (1-done) * 0.99 * s2_next_target)
            
            self.next_q_old_policy, self.next_s2_old_policy = self.critic(next_state, next_action, united=True)

        out = self.critic(state, action, united=False)
        critic_loss = (ReHE(q_value - out[0]) + ReHE(q_value - out[1]) + 
                      ReHE(q_value - out[2]) + ReHE(s2_value - out[3]))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_value)
        self.critic_optimizer.step()
        
        return critic_loss

    def actor_update(self, state, next_state):
        action = self.actor(state, mean=True)
        q_new_policy, s2_new_policy = self.critic(state, action, united=True)
        
        actor_loss = (-ReHaE(q_new_policy - self.q_old_policy) - 
                     ReHaE(s2_new_policy - self.s2_old_policy))

        next_action = self.actor(next_state, mean=True)
        next_q_new_policy, next_s2_new_policy = self.critic(next_state, next_action, united=True)
        actor_loss += (-ReHaE(next_q_new_policy - self.next_q_old_policy.mean().detach()) -
                      ReHaE(next_s2_new_policy - self.next_s2_old_policy.mean().detach()))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
        self.actor_optimizer.step()

        with torch.no_grad():
            self.q_old_policy = q_new_policy.mean().detach()
            self.s2_old_policy = s2_new_policy.mean().detach()

        return actor_loss

# Backward compatibility
class ReplayBuffer(EnhancedReplayBuffer):
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        super().__init__(state_dim, action_dim, capacity, device, fade_factor, stall_penalty)

# Enhanced noise with better exploration schedule
class EnhancedNoise:
    def __init__(self, max_action, burst=False, tr_noise=True):
        self.x_coor = 0.0
        self.tr_noise = tr_noise
        self.scale = 1.0 if burst else 0.15
        self.max_action = max_action
        
        # Enhanced noise parameters
        self.min_noise = 0.05
        self.exploration_phase_end = 1.5  # Extend exploration slightly

    def generate(self, x):
        if self.tr_noise and self.x_coor >= 2.133: 
            return (0.07*torch.randn_like(x)).clamp(-0.175, 0.175)
        if self.x_coor >= math.pi: 
            return torch.clamp(self.min_noise*torch.randn_like(x), -0.1, 0.1)
        
        with torch.no_grad():
            # Smoother noise decay
            decay_factor = max(0.1, math.cos(self.x_coor / self.exploration_phase_end) + 1.0)
            eps = self.scale * self.max_action * decay_factor
            lim = 2.5*eps
            self.x_coor += 3e-5
            return (eps*torch.randn_like(x)).clamp(-lim, lim)

# Keep original GANoise for backward compatibility
class GANoise(EnhancedNoise):
    def __init__(self, max_action, burst=False, tr_noise=True):
        super().__init__(max_action, burst, tr_noise)

class OUNoise:
    def __init__(self, action_dim, device, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.device = device
        self.x_coor = 0.0
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.ones(self.action_dim).to(self.device) * self.mu
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim).to(self.device) * self.mu

    def generate(self, x):
        if self.x_coor >= math.pi: 
            return 0.0
        with torch.no_grad():
            eps = (math.cos(self.x_coor) + 1.0)
            self.x_coor += 7e-4
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * torch.randn_like(x)
            self.state = x + dx
            return eps*self.state
