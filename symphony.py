# Enhanced Symphony algorithm with World Model, Hybrid Training, and Speed Optimizations
# Includes: Curiosity-driven exploration, prioritized replay, multi-step learning, 
# optimized architectures, and aggressive training schedules for maximum performance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
import random

# Set random seeds for reproducibility
torch.manual_seed(830143436)
np.random.seed(167430301)
random.seed(2193498338)

# Optimized loss functions for stability and speed
def stable_huber_loss(pred, target, delta=1.0):
    """Fast numerically stable Huber loss"""
    abs_error = torch.abs(pred - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)

def stable_mse_loss(pred, target):
    """Stable MSE with regularization"""
    return F.mse_loss(pred, target) + 0.005 * torch.mean((pred - target)**2)

def compute_n_step_returns(rewards, dones, next_values, gamma=0.99, n_steps=5):
    """Compute n-step returns for faster learning"""
    returns = torch.zeros_like(rewards)
    batch_size = rewards.shape[0]
    
    for i in range(batch_size):
        n_step_return = 0
        gamma_power = 1
        
        for step in range(min(n_steps, batch_size - i)):
            if i + step < batch_size:
                n_step_return += gamma_power * rewards[i + step]
                gamma_power *= gamma
                
                if dones[i + step]:
                    break
        
        # Add bootstrapped value if not terminated
        if i + n_steps < batch_size and not dones[i + n_steps - 1]:
            n_step_return += gamma_power * next_values[i + n_steps]
            
        returns[i] = n_step_return
    
    return returns

# Curiosity Module for Exploration
class CuriosityModule:
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        self.device = device
        
        # Inverse dynamics model - predicts action from state transitions
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        ).to(device)
        
        # Forward dynamics model - predicts next state from state-action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()),
            lr=1e-3, weight_decay=1e-5
        )
        
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute curiosity-based intrinsic reward"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            # Prediction error as curiosity signal
            pred_next_state = self.forward_model(torch.cat([state_t, action_t], dim=1))
            curiosity = torch.mean((pred_next_state - next_state_t) ** 2).item()
            
            return min(curiosity * 15.0, 2.0)  # Scaled and capped curiosity reward
    
    def train_curiosity(self, batch):
        """Train curiosity module on experience batch"""
        states, actions, _, next_states, _ = batch
        
        # Forward model loss - predict next state
        pred_next_states = self.forward_model(torch.cat([states, actions], dim=1))
        forward_loss = F.mse_loss(pred_next_states, next_states)
        
        # Inverse model loss - predict action from state transition
        pred_actions = self.inverse_model(torch.cat([states, next_states], dim=1))
        inverse_loss = F.mse_loss(pred_actions, actions)
        
        total_loss = forward_loss + 0.2 * inverse_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()), 1.0
        )
        self.optimizer.step()
        
        return total_loss.item()

# Fast activation functions
class ReSine(nn.Module):
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)

class FastFourierSeries(nn.Module):
    def __init__(self, hidden_dim, f_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ReSine(),
            nn.Dropout(0.05),  # Reduced dropout for speed
            nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x):
        return self.net(x)

# Optimized Fast Actor
class FastActor(nn.Module):
    def __init__(self, state_dim, action_dim, device, max_action=1.0, burst=False, tr_noise=True):
        super().__init__()
        
        # Smaller network for speed
        hidden_dim = 128
        
        self.input = nn.Linear(state_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = torch.mean(max_action).item() if hasattr(max_action, '__iter__') else max_action
        self.noise = OptimizedNoise(max_action, burst, tr_noise)
        self.device = device
        
    def forward(self, state, mean=False):
        x = self.input(state)
        x = self.max_action * self.net(x)
        
        if mean:
            return x
            
        x += self.noise.generate(x)
        return torch.clamp(x, -self.max_action, self.max_action)

# Optimized Fast Critic
class FastCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Optimized smaller network
        hidden_dim = 128
        
        self.input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Shared trunk with multiple heads for efficiency
        self.shared = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Q-network heads
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(4)
        ])
        
    def forward(self, state, action, united=False):
        x = torch.cat([state, action], dim=-1)
        x = self.layer_norm(self.input(x))
        features = self.shared(x)
        
        outputs = [head(features) for head in self.q_heads]
        
        if not united:
            return outputs
        
        # Return min Q-value and variance estimate
        q_min = torch.min(torch.stack(outputs[:3], dim=-1), dim=-1).values
        return q_min, outputs[3]

# Optimized Noise Generator
class OptimizedNoise:
    def __init__(self, max_action, burst=False, tr_noise=True):
        self.x_coor = 0.0
        self.tr_noise = tr_noise
        self.scale = 0.8 if burst else 0.2
        self.max_action = max_action
        self.min_noise = 0.03
        self.exploration_phase_end = 1.8
        
    def generate(self, x):
        # Fast noise generation with optimized schedule
        if self.tr_noise and self.x_coor >= 2.5:
            return 0.03 * torch.randn_like(x).clamp(-0.08, 0.08)
        
        if self.x_coor >= math.pi:
            return self.min_noise * torch.randn_like(x).clamp(-0.05, 0.05)
        
        # Optimized smooth decay
        with torch.no_grad():
            decay_factor = max(0.03, math.cos(self.x_coor / self.exploration_phase_end) + 1.0)
            eps = self.scale * self.max_action * decay_factor
            lim = 2.2 * eps
            self.x_coor += 4e-5
            return (eps * torch.randn_like(x)).clamp(-lim, lim)

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, alpha=0.6, beta=0.4):
        # Handle capacity specification
        capacity_dict = {"short": 80000, "medium": 200000, "full": 400000}
        if isinstance(capacity, str):
            self.capacity = capacity_dict[capacity]
        else:
            self.capacity = capacity
            
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.ptr = 0
        self.size = 0
        
        # Storage tensors
        self.states = torch.zeros((self.capacity, state_dim)).to(device)
        self.actions = torch.zeros((self.capacity, action_dim)).to(device)
        self.rewards = torch.zeros((self.capacity, 1)).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim)).to(device)
        self.dones = torch.zeros((self.capacity, 1)).to(device)
        
        # Priority management
        self.priorities = torch.ones(self.capacity).to(device)
        self.max_priority = 1.0
        
        # Adaptive batch size
        self.batch_size = 128
        
        # Normalization
        self.min_values = None
        self.max_values = None
        self.normalized = False
        
    def add(self, state, action, reward, next_state, done, td_error=1.0):
        index = self.ptr % self.capacity
        
        # Store transition
        self.states[index] = torch.FloatTensor(state).to(self.device)
        self.actions[index] = torch.FloatTensor(action).to(self.device)
        self.rewards[index] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[index] = torch.FloatTensor(next_state).to(self.device)
        self.dones[index] = torch.FloatTensor([done]).to(self.device)
        
        # Set priority
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities[index] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
        
        # Update batch size adaptively
        if self.size > 200:
            self.batch_size = min(max(128, self.size//400), 512)
    
    def find_min_max(self):
        """Calculate normalization parameters"""
        if self.size == 0:
            return
            
        actual_length = min(self.size, self.capacity)
        self.min_values = torch.min(self.states[:actual_length], dim=0).values
        self.max_values = torch.max(self.states[:actual_length], dim=0).values
        
        # Handle edge cases
        self.min_values[torch.isinf(self.min_values)] = -1e3
        self.max_values[torch.isinf(self.max_values)] = 1e3
        
        self.min_values = 2.0 * (torch.floor(10.0 * self.min_values) / 10.0).reshape(1, -1).to(self.device)
        self.max_values = 2.0 * (torch.ceil(10.0 * self.max_values) / 10.0).reshape(1, -1).to(self.device)
        self.normalized = True
    
    def normalize(self, state):
        """Normalize state"""
        if not self.normalized:
            return state
            
        state = torch.clamp(state, -1e3, 1e3)
        range_vals = self.max_values - self.min_values
        range_vals[range_vals == 0] = 1.0
        state = 4.0 * (state - self.min_values) / range_vals - 2.0
        state[torch.isnan(state)] = 0.0
        return state
    
    def sample(self):
        """Sample with priority weighting"""
        if self.size == 0:
            return None, None, None
            
        # Compute sampling probabilities
        valid_priorities = self.priorities[:self.size]
        probs = valid_priorities / torch.sum(valid_priorities)
        
        # Sample indices
        batch_size = min(self.batch_size, self.size)
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / torch.max(weights)
        
        batch = (
            self.normalize(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self.normalize(self.next_states[indices]),
            self.dones[indices]
        )
        
        return batch, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size

# World Model for Dreaming (simplified for speed)
class FastWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Compact world model
        self.input_dim = state_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1 + 1)  # next_state + reward + done
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        
        next_state = out[:, :-2]
        reward = out[:, -2:-1]
        done = torch.sigmoid(out[:, -1:])
        
        return next_state, reward, done
    
    def train_step(self, batch):
        """Fast world model training"""
        states, actions, rewards, next_states, dones = batch
        
        pred_next_states, pred_rewards, pred_dones = self.forward(states, actions)
        
        # Simple losses for speed
        state_loss = F.mse_loss(pred_next_states, next_states)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        done_loss = F.binary_cross_entropy(pred_dones, dones)
        
        loss = state_loss + 0.5 * reward_loss + 0.1 * done_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()

# Ultra-Fast Symphony with All Optimizations
class UltraFastSymphony:
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, burst=False, tr_noise=True):
        # Optimized components
        self.actor = FastActor(state_dim, action_dim, device, max_action, burst, tr_noise).to(device)
        self.critic = FastCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Fast optimizers with higher learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4, weight_decay=1e-6)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-6)
        
        # World model and curiosity
        self.world_model = FastWorldModel(state_dim, action_dim, hidden_dim//4).to(device)
        self.curiosity = CuriosityModule(state_dim, action_dim, hidden_dim, device)
        
        # Training management
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Fast learning parameters
        self.world_model_ready = False
        self.world_model_loss_threshold = 0.1  # More lenient threshold
        self.dream_ratio = 0.0
        self.max_dream_ratio = 0.4
        
        # Training tracking
        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0
        self.tr_step = 0
        self.world_model_losses = []
        
    def select_action(self, state, replay_buffer=None, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1, self.state_dim).to(self.device)
            if replay_buffer and hasattr(replay_buffer, 'normalize'):
                state = replay_buffer.normalize(state)
            action = self.actor(state, mean=mean)
            return action.cpu().data.numpy().flatten()
    
    def train_world_model(self, replay_buffer, n_epochs=15):
        """Fast world model training"""
        if len(replay_buffer) < 500:
            return []
        
        losses = []
        for epoch in range(n_epochs):
            if hasattr(replay_buffer, 'sample'):
                sample_result = replay_buffer.sample()
                if sample_result is None:
                    continue
                    
                if len(sample_result) == 3:  # Prioritized buffer
                    batch, _, _ = sample_result
                else:  # Regular batch
                    batch = sample_result
            else:
                continue
                
            loss = self.world_model.train_step(batch)
            losses.append(loss)
        
        if losses:
            avg_loss = np.mean(losses[-5:])  # Last 5 epochs
            self.world_model_losses.append(avg_loss)
            
            # Aggressive readiness threshold
            if avg_loss < self.world_model_loss_threshold:
                self.world_model_ready = True
                self.dream_ratio = 0.1  # Start dreaming immediately
                print(f"World model ready! Loss: {avg_loss:.4f}")
        
        return losses
    
    def generate_dreams(self, real_batch, n_dreams=25, dream_length=8):
        """Fast dream generation"""
        if not self.world_model_ready:
            return None
            
        states, actions, rewards, next_states, dones = real_batch
        batch_size = min(n_dreams, states.shape[0])
        
        # Sample initial states
        init_indices = torch.randint(0, states.shape[0], (batch_size,))
        current_state = states[init_indices]
        
        # Generate short rollouts
        dream_states = [current_state]
        dream_actions = []
        dream_rewards = []
        dream_dones = []
        
        for step in range(dream_length):
            with torch.no_grad():
                action = self.actor(current_state, mean=True)
                next_state, reward, done = self.world_model(current_state, action)
                
                # Add small noise for diversity
                next_state += 0.01 * torch.randn_like(next_state)
                
                dream_actions.append(action)
                dream_rewards.append(reward)
                dream_dones.append(done)
                dream_states.append(next_state)
                
                current_state = next_state
                
                # Early termination
                if torch.any(done > 0.5):
                    break
        
        if len(dream_actions) == 0:
            return None
        
        # Format as transitions
        flat_states = torch.cat(dream_states[:-1], dim=0)
        flat_next_states = torch.cat(dream_states[1:], dim=0)
        flat_actions = torch.cat(dream_actions, dim=0)
        flat_rewards = torch.cat(dream_rewards, dim=0)
        flat_dones = torch.cat(dream_dones, dim=0)
        
        return (flat_states, flat_actions, flat_rewards, flat_next_states, flat_dones)
    
    def train(self, real_batch, dream_batch=None):
        """Ultra-fast training with all optimizations"""
        self.tr_step += 1
        
        # Determine training weights
        real_weight = 1.0 - self.dream_ratio
        dream_weight = self.dream_ratio
        
        # Train on real data
        real_loss = self.train_step(real_batch, weight=real_weight)
        
        total_loss = real_loss
        
        # Train on dreams
        if dream_batch is not None and dream_weight > 0:
            dream_loss = self.train_step(dream_batch, weight=dream_weight)
            total_loss += dream_loss
        
        # Update dream ratio aggressively
        if self.world_model_ready:
            self.dream_ratio = min(self.max_dream_ratio, self.dream_ratio + 0.002)
        
        return total_loss
    
    def train_step(self, batch, weight=1.0):
        """Optimized training step with n-step returns"""
        states, actions, rewards, next_states, dones = batch
        
        # Device placement
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if states.size(0) == 0:
            return torch.tensor(0.0)
        
        # Fast target update
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), 
                                         self.critic.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
            
            next_actions = self.actor(next_states, mean=True)
            q_next_target, _ = self.critic_target(next_states, next_actions, united=True)
            
            # Use 3-step returns for faster learning
            n_step_targets = compute_n_step_returns(rewards, dones, q_next_target, n_steps=3)
        
        # Critic update with n-step targets
        current_qs = self.critic(states, actions, united=False)
        
        critic_loss = sum(stable_mse_loss(q, n_step_targets) for q in current_qs[:3])
        critic_loss += stable_mse_loss(current_qs[3], 0.01 * torch.ones_like(rewards))
        critic_loss *= weight
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Actor update
        current_actions = self.actor(states, mean=True)
        q_new_policy, s2_new_policy = self.critic(states, current_actions, united=True)
        
        actor_loss = -torch.mean(q_new_policy) * weight
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update tracking
        with torch.no_grad():
            self.q_old_policy = torch.mean(q_new_policy).item()
            self.s2_old_policy = torch.mean(s2_new_policy).item()
        
        return critic_loss + actor_loss

# Adaptive training frequency function
def get_adaptive_training_freq(episode, buffer_size):
    """Dynamic training frequency based on data availability"""
    if buffer_size < 1000:
        return 1
    elif buffer_size < 10000:
        return min(6, buffer_size // 200)
    else:
        return min(12, buffer_size // 800)

# Backward compatibility classes
class Symphony(UltraFastSymphony):
    """Backward compatible Symphony class"""
    pass

class ReplayBuffer(PrioritizedReplayBuffer):
    """Backward compatible ReplayBuffer class"""
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        super().__init__(state_dim, action_dim, capacity, device)

class EnhancedReplayBuffer(PrioritizedReplayBuffer):
    """Enhanced buffer alias"""
    pass

# Legacy classes
class Actor(FastActor):
    pass

class Critic(FastCritic):
    pass

class EnhancedNoise(OptimizedNoise):
    pass

class GANoise(OptimizedNoise):
    pass

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
            return torch.zeros_like(x)
        
        with torch.no_grad():
            eps = (math.cos(self.x_coor) + 1.0)
            self.x_coor += 1e-3
            
            dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn_like(self.state)
            self.state = self.state + dx
            
            return eps * self.state
