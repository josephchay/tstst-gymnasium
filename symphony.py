# Enhanced Symphony algorithm with World Model and Hybrid Training
# Addresses all critical issues: proper sequential modeling, stable losses, 
# ensemble uncertainty, curriculum learning, and computational efficiency

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

# Stable loss functions (replaces unstable ReHE/ReHaE)
def stable_huber_loss(pred, target, delta=1.0):
    """Numerically stable Huber loss"""
    abs_error = torch.abs(pred - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)

def stable_mse_loss(pred, target):
    """Stable MSE with gradient clipping"""
    error = pred - target
    return F.mse_loss(pred, target) + 0.01 * torch.mean(error**2)

# Improved activation function
class ReSine(nn.Module):
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)

class FourierSeries(nn.Module):
    def __init__(self, hidden_dim, f_out):
        super().__init__()
        self.fft = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ReSine(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x):
        return self.fft(x)

# World Model Core Architecture
class WorldModelCore(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = state_dim + action_dim
        
        # GRU for sequential modeling (more stable than LSTM)
        self.rnn = nn.GRU(self.input_dim, hidden_dim, num_layers=2, 
                         batch_first=True, dropout=0.1)
        
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state_action_seq, hidden=None):
        rnn_out, hidden = self.rnn(state_action_seq, hidden)
        
        states = self.state_head(rnn_out)
        rewards = self.reward_head(rnn_out)
        dones = torch.sigmoid(self.done_head(rnn_out))
        
        return states, rewards, dones, hidden

# Ensemble World Model for Uncertainty Quantification
class EnsembleWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.models = nn.ModuleList([
            WorldModelCore(state_dim, action_dim, hidden_dim) 
            for _ in range(n_models)
        ])
        
        self.optimizers = [optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 
                          for model in self.models]
        
    def forward(self, state_action_seq):
        predictions = []
        for model in self.models:
            states, rewards, dones, _ = model(state_action_seq)
            predictions.append((states, rewards, dones))
        
        # Stack predictions for ensemble statistics
        states_stack = torch.stack([p[0] for p in predictions])
        rewards_stack = torch.stack([p[1] for p in predictions])
        dones_stack = torch.stack([p[2] for p in predictions])
        
        # Compute ensemble mean and variance
        state_mean = states_stack.mean(0)
        state_var = states_stack.var(0) + 1e-6  # Small epsilon for numerical stability
        reward_mean = rewards_stack.mean(0)
        done_mean = dones_stack.mean(0)
        
        return state_mean, reward_mean, done_mean, state_var
    
    def rollout(self, init_state, actions, max_uncertainty=0.1):
        """Generate multi-step rollouts with uncertainty filtering"""
        batch_size, seq_len = actions.shape[:2]
        
        states_list = []
        rewards_list = []
        dones_list = []
        current_state = init_state
        
        for t in range(seq_len):
            # Create state-action input
            state_action = torch.cat([current_state, actions[:, t]], dim=-1).unsqueeze(1)
            
            # Get ensemble prediction
            next_state, reward, done, uncertainty = self.forward(state_action)
            next_state = next_state.squeeze(1)
            reward = reward.squeeze(1)
            done = done.squeeze(1)
            uncertainty = uncertainty.squeeze(1)
            
            # Filter by uncertainty - stop rollout if too uncertain
            avg_uncertainty = torch.mean(uncertainty, dim=-1)
            if torch.any(avg_uncertainty > max_uncertainty):
                break
                
            states_list.append(next_state)
            rewards_list.append(reward)
            dones_list.append(done)
            current_state = next_state
            
            # Stop if episode ends
            if torch.any(done > 0.5):
                break
        
        if not states_list:
            return None, None, None
            
        return (torch.stack(states_list, 1), 
                torch.stack(rewards_list, 1), 
                torch.stack(dones_list, 1))
    
    def train_step(self, batch):
        """Train all models in ensemble"""
        state_action_seq, target_states, target_rewards, target_dones = batch
        
        total_loss = 0
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            pred_states, pred_rewards, pred_dones, _ = model(state_action_seq)
            
            # Stable losses
            state_loss = F.mse_loss(pred_states, target_states)
            reward_loss = F.mse_loss(pred_rewards.squeeze(-1), target_rewards.squeeze(-1))
            done_loss = F.binary_cross_entropy(pred_dones.squeeze(-1), target_dones.squeeze(-1))
            
            loss = state_loss + 0.5 * reward_loss + 0.1 * done_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / self.n_models

# Sequential Replay Buffer for World Model Training
class SequentialReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, seq_len=15):
        self.capacity = capacity_dict = {"short": 50000, "medium": 150000, "full": 300000}
        if isinstance(capacity, str):
            self.capacity = capacity_dict[capacity]
        else:
            self.capacity = capacity
            
        self.seq_len = seq_len
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Store individual transitions
        self.states = torch.zeros((self.capacity, state_dim)).to(device)
        self.actions = torch.zeros((self.capacity, action_dim)).to(device)
        self.rewards = torch.zeros((self.capacity, 1)).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim)).to(device)
        self.dones = torch.zeros((self.capacity, 1)).to(device)
        self.episode_ids = torch.zeros(self.capacity, dtype=torch.long).to(device)
        
        self.current_episode_id = 0
        self.batch_size = 128
        
        # Normalization parameters
        self.min_values = None
        self.max_values = None
        self.normalized = False
        
    def add(self, state, action, reward, next_state, done):
        index = self.ptr % self.capacity
        
        self.states[index] = torch.FloatTensor(state).to(self.device)
        self.actions[index] = torch.FloatTensor(action).to(self.device)
        self.rewards[index] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[index] = torch.FloatTensor(next_state).to(self.device)
        self.dones[index] = torch.FloatTensor([done]).to(self.device)
        self.episode_ids[index] = self.current_episode_id
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
        
        if done:
            self.current_episode_id += 1
            
        # Update batch size
        if self.size > 100:
            self.batch_size = min(max(128, self.size//500), 1024)
    
    def find_min_max(self):
        """Calculate normalization parameters"""
        if self.size == 0:
            return
            
        actual_length = min(self.size, self.capacity)
        self.min_values = torch.min(self.states[:actual_length], dim=0).values
        self.max_values = torch.max(self.states[:actual_length], dim=0).values
        
        # Handle inf values
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
        """Sample individual transitions"""
        if self.size < self.batch_size:
            indices = torch.arange(self.size)
        else:
            indices = torch.randint(0, self.size, (self.batch_size,))
        
        return (
            self.normalize(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self.normalize(self.next_states[indices]),
            self.dones[indices]
        )
    
    def sample_sequences(self, batch_size=32):
        """Sample sequences for world model training"""
        if self.size < self.seq_len * 2:
            return None
            
        # Find valid sequence starting points
        valid_starts = []
        for i in range(self.size - self.seq_len):
            if (self.episode_ids[i] == self.episode_ids[i + self.seq_len - 1] and
                torch.all(self.dones[i:i+self.seq_len-1] < 0.5)):
                valid_starts.append(i)
        
        if len(valid_starts) < batch_size:
            return None
        
        # Sample starting indices
        start_indices = np.random.choice(valid_starts, batch_size, replace=False)
        
        # Build sequences
        state_action_seqs = []
        target_states = []
        target_rewards = []
        target_dones = []
        
        for start_idx in start_indices:
            end_idx = start_idx + self.seq_len
            
            # State-action sequence
            states_seq = self.normalize(self.states[start_idx:end_idx])
            actions_seq = self.actions[start_idx:end_idx]
            state_action_seq = torch.cat([states_seq, actions_seq], dim=-1)
            
            # Targets
            next_states_seq = self.normalize(self.next_states[start_idx:end_idx])
            rewards_seq = self.rewards[start_idx:end_idx]
            dones_seq = self.dones[start_idx:end_idx]
            
            state_action_seqs.append(state_action_seq)
            target_states.append(next_states_seq)
            target_rewards.append(rewards_seq)
            target_dones.append(dones_seq)
        
        return (
            torch.stack(state_action_seqs),
            torch.stack(target_states),
            torch.stack(target_rewards),
            torch.stack(target_dones)
        )
    
    def __len__(self):
        return self.size

# Fixed Actor with Stable Exploration
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=256, max_action=1.0, burst=False, tr_noise=True):
        super(Actor, self).__init__()
        
        self.input = nn.Linear(state_dim, hidden_dim)
        self.net = nn.Sequential(
            FourierSeries(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = torch.mean(max_action).item() if hasattr(max_action, '__iter__') else max_action
        self.noise = StableNoise(max_action, burst, tr_noise)
        self.device = device
        
    def forward(self, state, mean=False):
        x = self.input(state)
        x = self.max_action * self.net(x)
        
        if mean:
            return x
            
        x += self.noise.generate(x)
        return torch.clamp(x, -self.max_action, self.max_action)

# Fixed Critic with Stable Architecture
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Three Q-networks + variance estimator
        networks = []
        for _ in range(4):
            net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )
            networks.append(net)
        
        self.networks = nn.ModuleList(networks)
        
    def forward(self, state, action, united=False):
        x = torch.cat([state, action], dim=-1)
        x = self.layer_norm(self.input(x))
        
        outputs = [net(x) for net in self.networks]
        
        if not united:
            return outputs
        
        # Return min Q-value and variance estimate
        q_min = torch.min(torch.stack(outputs[:3], dim=-1), dim=-1).values
        return q_min, outputs[3]

# Stable Noise Generator
class StableNoise:
    def __init__(self, max_action, burst=False, tr_noise=True):
        self.x_coor = 0.0
        self.tr_noise = tr_noise
        self.scale = 0.8 if burst else 0.15
        self.max_action = max_action
        self.min_noise = 0.05
        self.exploration_phase_end = 2.0
        
    def generate(self, x):
        if self.tr_noise and self.x_coor >= 2.5:
            return 0.05 * torch.randn_like(x).clamp(-0.1, 0.1)
        
        if self.x_coor >= math.pi:
            return self.min_noise * torch.randn_like(x).clamp(-0.1, 0.1)
        
        # Smooth decay
        with torch.no_grad():
            decay_factor = max(0.05, math.cos(self.x_coor / self.exploration_phase_end) + 1.0)
            eps = self.scale * self.max_action * decay_factor
            lim = 2.0 * eps
            self.x_coor += 2e-5
            return (eps * torch.randn_like(x)).clamp(-lim, lim)

# Enhanced Noise (backward compatibility)
class EnhancedNoise(StableNoise):
    pass

class GANoise(StableNoise):
    pass

# Hybrid Symphony with World Model Integration
class HybridSymphony:
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, burst=False, tr_noise=True):
        # Core Symphony components
        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action, burst, tr_noise).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-5)
        
        # World model components
        self.world_model = EnsembleWorldModel(state_dim, action_dim, hidden_dim).to(device)
        
        # Training management
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # World model tracking
        self.world_model_loss_history = []
        self.world_model_ready = False
        self.model_accuracy_threshold = 0.05
        
        # Curriculum learning
        self.dream_ratio = 0.0
        self.max_dream_ratio = 0.3
        self.curriculum_step = 0.01
        
        # Symphony state tracking
        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0
        self.tr_step = 0
        
    def select_action(self, state, replay_buffer=None, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1, self.state_dim).to(self.device)
            if replay_buffer and hasattr(replay_buffer, 'normalize'):
                state = replay_buffer.normalize(state)
            action = self.actor(state, mean=mean)
            return action.cpu().data.numpy().flatten()
    
    def train_world_model(self, seq_buffer, n_epochs=50):
        """Train world model ensemble"""
        print("Training world model ensemble...")
        
        epoch_losses = []
        for epoch in range(n_epochs):
            batch = seq_buffer.sample_sequences(32)
            if batch is None:
                continue
                
            loss = self.world_model.train_step(batch)
            epoch_losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses[-10:])
            self.world_model_loss_history.append(avg_loss)
            
            if avg_loss < self.model_accuracy_threshold:
                self.world_model_ready = True
                print(f"World model ready! Final loss: {avg_loss:.6f}")
            else:
                print(f"World model needs more training. Loss: {avg_loss:.6f}")
        
        return epoch_losses
    
    def update_curriculum(self):
        """Update dream ratio based on model performance"""
        if not self.world_model_ready or not self.world_model_loss_history:
            self.dream_ratio = 0.0
            return
        
        recent_loss = self.world_model_loss_history[-1]
        
        if recent_loss < self.model_accuracy_threshold * 0.5:
            self.dream_ratio = min(self.max_dream_ratio, 
                                 self.dream_ratio + self.curriculum_step)
        elif recent_loss > self.model_accuracy_threshold * 2:
            self.dream_ratio = max(0.0, self.dream_ratio - self.curriculum_step * 2)
        
        print(f"Dream ratio: {self.dream_ratio:.3f}, Model loss: {recent_loss:.6f}")
    
    def generate_dreams(self, real_batch, n_dreams=30, dream_length=12):
        """Generate synthetic rollouts"""
        if not self.world_model_ready or self.dream_ratio == 0:
            return None
        
        states, actions, rewards, next_states, dones = real_batch
        batch_size = min(n_dreams, states.shape[0])
        
        # Sample initial states
        init_indices = torch.randint(0, states.shape[0], (batch_size,))
        init_states = states[init_indices]
        
        # Generate actions using current policy
        dream_actions = []
        current_state = init_states
        
        for _ in range(dream_length):
            with torch.no_grad():
                action = self.actor(current_state, mean=True)
                dream_actions.append(action)
                # Add small noise to prevent deterministic rollouts
                current_state = current_state + 0.01 * torch.randn_like(current_state)
        
        dream_actions = torch.stack(dream_actions, dim=1)
        
        # Generate rollouts
        dream_states, dream_rewards, dream_dones = self.world_model.rollout(
            init_states, dream_actions, max_uncertainty=0.1
        )
        
        if dream_states is None:
            return None
        
        # Format as transitions
        seq_len = dream_states.shape[1]
        
        # Flatten sequences
        flat_states = dream_states[:, :-1].reshape(-1, self.state_dim)
        flat_next_states = dream_states[:, 1:].reshape(-1, self.state_dim)
        flat_actions = dream_actions[:, :seq_len-1].reshape(-1, self.action_dim)
        flat_rewards = dream_rewards[:, :seq_len-1].reshape(-1, 1)
        flat_dones = dream_dones[:, :seq_len-1].reshape(-1, 1)
        
        return (flat_states, flat_actions, flat_rewards, flat_next_states, flat_dones)
    
    def train(self, real_batch, dream_batch=None):
        """Hybrid training with curriculum learning"""
        self.tr_step += 1
        
        # Always train on real data
        real_loss = self.train_symphony(real_batch, weight=1.0 - self.dream_ratio)
        
        total_loss = real_loss
        
        # Train on dreams if available
        if dream_batch is not None and self.dream_ratio > 0:
            dream_loss = self.train_symphony(dream_batch, weight=self.dream_ratio)
            total_loss = real_loss + dream_loss
        
        return total_loss
    
    def train_symphony(self, batch, weight=1.0):
        """Core Symphony training with stable losses"""
        states, actions, rewards, next_states, dones = batch
        
        # Ensure proper device placement
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if states.size(0) == 0:
            return torch.tensor(0.0)
        
        # Critic update
        with torch.no_grad():
            # Soft target update
            for target_param, param in zip(self.critic_target.parameters(), 
                                         self.critic.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
            
            next_actions = self.actor(next_states, mean=True)
            q_next_target, s2_next_target = self.critic_target(next_states, next_actions, united=True)
            q_target = rewards + (1 - dones) * 0.99 * q_next_target
            s2_target = 0.01 * torch.ones_like(rewards)
        
        # Current Q-values
        current_qs = self.critic(states, actions, united=False)
        
        # Stable critic loss
        critic_loss = sum(stable_mse_loss(q, q_target) for q in current_qs[:3])
        critic_loss += stable_mse_loss(current_qs[3], s2_target)
        critic_loss *= weight
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor update
        current_actions = self.actor(states, mean=True)
        q_new_policy, s2_new_policy = self.critic(states, current_actions, united=True)
        
        actor_loss = -torch.mean(q_new_policy) * weight
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update tracking
        with torch.no_grad():
            self.q_old_policy = torch.mean(q_new_policy).item()
            self.s2_old_policy = torch.mean(s2_new_policy).item()
        
        return critic_loss + actor_loss

# Backward compatibility classes
class Symphony(HybridSymphony):
    """Backward compatible Symphony class"""
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, burst=False, tr_noise=True):
        super().__init__(state_dim, action_dim, hidden_dim, device, max_action, burst, tr_noise)

class ReplayBuffer(SequentialReplayBuffer):
    """Backward compatible ReplayBuffer class"""
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        super().__init__(state_dim, action_dim, capacity, device)
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty

# Legacy noise classes for compatibility
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
            self.x_coor += 7e-4
            
            dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn_like(self.state)
            self.state = self.state + dx
            
            return eps * self.state
