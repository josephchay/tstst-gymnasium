#This is the enhanced version of Symphony algorithm with model-based dreaming capabilities.
#Implements uncertainty-aware model selection and balanced real/dream data training.

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

# FeedForward Transformer for Environment Model
class FeedForwardTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, ensemble_size=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        
        # Input embedding
        self.input_emb = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        # Ensemble of models for uncertainty estimation
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                ReSine(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                ReSine(),
                nn.Linear(hidden_dim, state_dim + 1)  # next_state + reward
            ) for _ in range(ensemble_size)
        ])
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.layer_norm1(self.input_emb(x))
        
        # Get predictions from all ensemble models
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions, dim=0)  # [ensemble_size, batch, state_dim+1]
        
        # Split into next_state and reward predictions
        next_states = predictions[:, :, :-1]  # [ensemble_size, batch, state_dim]
        rewards = predictions[:, :, -1:]      # [ensemble_size, batch, 1]
        
        # Calculate mean and uncertainty
        next_state_mean = next_states.mean(dim=0)
        reward_mean = rewards.mean(dim=0)
        
        next_state_uncertainty = next_states.var(dim=0).mean(dim=-1, keepdim=True)
        reward_uncertainty = rewards.var(dim=0)
        
        return next_state_mean, reward_mean, next_state_uncertainty, reward_uncertainty

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0, burst=False, tr_noise=True):
        super(Actor, self).__init__()
        self.input = nn.Linear(state_dim, hidden_dim)
        self.net = nn.Sequential(
            FourierSeries(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = torch.mean(max_action).item()
        self.noise = GANoise(max_action, burst, tr_noise)

    def forward(self, state, mean=False):
        x = self.input(state)
        x = self.max_action*self.net(x)
        if mean: return x
        x += self.noise.generate(x)
        return x.clamp(-self.max_action, self.max_action)

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()
        self.input = nn.Linear(state_dim+action_dim, hidden_dim)
        qA = FourierSeries(hidden_dim, 1)
        qB = FourierSeries(hidden_dim, 1)
        qC = FourierSeries(hidden_dim, 1)
        s2 = FourierSeries(hidden_dim, 1)
        self.nets = nn.ModuleList([qA, qB, qC, s2])

    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        x = self.input(x)
        xs = [net(x) for net in self.nets]
        if not united: return xs
        qmin = torch.min(torch.stack(xs[:3], dim=-1), dim=-1).values
        return qmin, xs[3]

# Enhanced Replay Buffer with real/dream data balancing
class EnhancedReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        capacity_dict = {"short": 100000, "medium": 300000, "full": 500000}
        self.capacity, self.length, self.device = capacity_dict[capacity], 0, device
        self.batch_size = min(max(128, self.length//500), 1024)
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs, self.step = [], np.array([]), np.array([]), 0
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty
        
        # Separate storage for real vs dreamed data
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.is_real = torch.ones((self.capacity, 1), dtype=torch.bool).to(device)  # Track real vs dreamed
        
        # Data balancing parameters
        self.real_data_ratio = 0.7  # Maintain 70% real data
        self.min_real_ratio = 0.5   # Never go below 50% real data
        
        self.raw = True

    def find_min_max(self):
        self.min_values = torch.min(self.states, dim=0).values
        self.max_values = torch.max(self.states, dim=0).values
        self.min_values[torch.isinf(self.min_values)] = -1e+3
        self.max_values[torch.isinf(self.max_values)] = 1e+3
        self.min_values = 2.0*(torch.floor(10.0*self.min_values)/10.0).reshape(1, -1).to(self.device)
        self.max_values = 2.0*(torch.ceil(10.0*self.max_values)/10.0).reshape(1, -1).to(self.device)
        self.raw = False

    def normalize(self, state):
        if self.raw: return state
        state[torch.isneginf(state)] = -1e+3
        state[torch.isposinf(state)] = 1e+3
        state = 4.0 * (state - self.min_values) / ((self.max_values - self.min_values)) - 2.0
        state[torch.isnan(state)] = 0.0
        return state

    def add(self, state, action, reward, next_state, done, is_real=True):
        if self.length < self.capacity:
            self.length += 1
        
        index = self.step % self.capacity
        self.states[index] = torch.FloatTensor(state).to(self.device)
        self.actions[index] = torch.FloatTensor(action).to(self.device)
        self.rewards[index] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[index] = torch.FloatTensor(next_state).to(self.device)
        self.dones[index] = torch.FloatTensor([done]).to(self.device)
        self.is_real[index] = torch.tensor([is_real], dtype=torch.bool).to(self.device)
        
        self.step += 1
        if self.length > 100: 
            self.batch_size = min(max(128, self.length//500), 1024)

    def get_real_dream_indices(self):
        """Get indices for real and dreamed data"""
        if self.length == 0:
            return np.array([]), np.array([])
        
        current_indices = np.arange(min(self.length, self.capacity))
        is_real_np = self.is_real[:self.length].cpu().numpy().flatten()
        
        real_indices = current_indices[is_real_np]
        dream_indices = current_indices[~is_real_np]
        
        return real_indices, dream_indices

    def balanced_sample(self):
        """Sample with balanced real/dream ratio"""
        real_indices, dream_indices = self.get_real_dream_indices()
        
        if len(dream_indices) == 0:
            # No dream data, sample normally
            return self.sample(uniform=False)
        
        # Calculate target numbers
        n_real = max(int(self.batch_size * self.real_data_ratio), 
                    min(len(real_indices), int(self.batch_size * self.min_real_ratio)))
        n_dream = min(self.batch_size - n_real, len(dream_indices))
        
        # Sample from each type
        sampled_real = self.random.choice(real_indices, size=n_real, replace=True) if n_real > 0 else []
        sampled_dream = self.random.choice(dream_indices, size=n_dream, replace=True) if n_dream > 0 else []
        
        # Combine indices
        all_indices = np.concatenate([sampled_real, sampled_dream]) if n_dream > 0 else sampled_real
        
        return (
            self.normalize(self.states[all_indices]),
            self.actions[all_indices],
            self.rewards[all_indices],
            self.normalize(self.next_states[all_indices]),
            self.dones[all_indices]
        )

    def generate_probs(self, uniform=False):
        if uniform or self.length <= 100: 
            self.indexes = np.arange(self.length)
            return None

        if self.length >= self.capacity: return self.probs

        def fade(norm_index): return np.tanh(self.fade_factor*norm_index**2)
        
        self.indexes = np.arange(self.length)
        weights = 1e-7*(fade(self.indexes/self.length))
        self.probs = weights/np.sum(weights)
        return self.probs

    def sample(self, uniform=False):
        indices = self.random.choice(self.indexes, p=self.generate_probs(uniform), size=self.batch_size)
        return (
            self.normalize(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self.normalize(self.next_states[indices]),
            self.dones[indices]
        )

    def __len__(self):
        return self.length

# Enhanced Symphony with model-based dreaming
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, burst=False, tr_noise=True):
        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action, burst, tr_noise).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Add environment model
        self.env_model = FeedForwardTransformer(state_dim, action_dim, hidden_dim//2).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)
        self.model_optimizer = optim.Adam(self.env_model.parameters(), lr=1e-3)
        
        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0
        self.tr_step = 0
        
        # Model-based parameters
        self.model_rollout_length = 3  # Start conservative
        self.max_rollout_length = 7    # Conservative maximum
        self.uncertainty_threshold = 0.1
        self.model_train_freq = 4
        self.use_model = False
        self.model_usage_ratio = 0.0
        
    def train_model(self, replay_buffer):
        """Train the environment model"""
        if len(replay_buffer) < 1000:
            return
            
        batch = replay_buffer.sample(uniform=True)
        state, action, reward, next_state, done = batch
        
        pred_next_state, pred_reward, state_uncertainty, reward_uncertainty = self.env_model(state, action)
        
        # Model loss
        state_loss = F.mse_loss(pred_next_state, next_state)
        reward_loss = F.mse_loss(pred_reward, reward)
        model_loss = state_loss + reward_loss
        
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        
        # Update model usage based on accuracy
        with torch.no_grad():
            avg_uncertainty = (state_uncertainty.mean() + reward_uncertainty.mean()) / 2
            if avg_uncertainty < self.uncertainty_threshold:
                self.use_model = True
                self.model_usage_ratio = min(0.3, self.model_usage_ratio + 0.01)
            else:
                self.model_usage_ratio = max(0.0, self.model_usage_ratio - 0.02)

    def generate_model_data(self, replay_buffer, n_rollouts=50):
        """Generate synthetic data using the environment model"""
        if not self.use_model or len(replay_buffer) < 1000:
            return
            
        with torch.no_grad():
            # Sample initial states from replay buffer
            batch = replay_buffer.sample(uniform=True)
            init_states = batch[0][:n_rollouts//2]  # Use half batch size for initial states
            
            for rollout_step in range(self.model_rollout_length):
                actions = self.actor(init_states, mean=True)
                pred_next_states, pred_rewards, uncertainties, _ = self.env_model(init_states, actions)
                
                # Only use predictions with low uncertainty
                low_uncertainty_mask = uncertainties.squeeze() < self.uncertainty_threshold
                if low_uncertainty_mask.sum() == 0:
                    break
                
                # Add synthetic transitions to replay buffer
                for i in range(len(init_states)):
                    if low_uncertainty_mask[i]:
                        replay_buffer.add(
                            init_states[i].cpu().numpy(),
                            actions[i].cpu().numpy(), 
                            pred_rewards[i].item(),
                            pred_next_states[i].cpu().numpy(),
                            False,  # Not terminal for model rollouts
                            is_real=False  # Mark as synthetic data
                        )
                
                # Continue rollout from predicted states
                init_states = pred_next_states[low_uncertainty_mask]
                if len(init_states) == 0:
                    break

    def select_action(self, state, replay_buffer=None, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1, self.state_dim).to(self.device)
            if replay_buffer: state = replay_buffer.normalize(state)
            action = self.actor(state, mean=mean)
            return action.cpu().data.numpy().flatten()

    def train(self, batch, replay_buffer=None):
        self.tr_step += 1
        
        # Train environment model periodically
        if replay_buffer and self.tr_step % self.model_train_freq == 0:
            self.train_model(replay_buffer)
            
        # Generate synthetic data periodically  
        if replay_buffer and self.use_model and self.tr_step % 50 == 0:
            self.generate_model_data(replay_buffer)
        
        state, action, reward, next_state, done = batch
        self.critic_update(state, action, reward, next_state, done)
        return self.actor_update(state, next_state)

    def critic_update(self, state, action, reward, next_state, done):
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)
            
            next_action = self.actor(next_state, mean=True)
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward + (1-done) * 0.99 * q_next_target
            s2_value = 3e-3 * (3e-3 * torch.var(reward) + (1-done) * 0.99 * s2_next_target)
            
            self.next_q_old_policy, self.next_s2_old_policy = self.critic(next_state, next_action, united=True)
        
        out = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - out[0]) + ReHE(q_value - out[1]) + ReHE(q_value - out[2]) + ReHE(s2_value - out[3])
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def actor_update(self, state, next_state):
        action = self.actor(state, mean=True)
        q_new_policy, s2_new_policy = self.critic(state, action, united=True)
        
        actor_loss = -ReHaE(q_new_policy - self.q_old_policy) -ReHaE(s2_new_policy - self.s2_old_policy)
        
        next_action = self.actor(next_state, mean=True)
        next_q_new_policy, next_s2_new_policy = self.critic(next_state, next_action, united=True)
        actor_loss += -ReHaE(next_q_new_policy - self.next_q_old_policy.mean().detach()) -ReHaE(next_s2_new_policy - self.next_s2_old_policy.mean().detach())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        with torch.no_grad():
            self.q_old_policy = q_new_policy.mean().detach()
            self.s2_old_policy = s2_new_policy.mean().detach()
        
        return actor_loss

# Keep original ReplayBuffer for backward compatibility
class ReplayBuffer(EnhancedReplayBuffer):
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        super().__init__(state_dim, action_dim, capacity, device, fade_factor, stall_penalty)

# NOISES with cosine decrease
class GANoise:
    def __init__(self, max_action, burst=False, tr_noise=True):
        self.x_coor = 0.0
        self.tr_noise = tr_noise
        self.scale = 1.0 if burst else 0.15
        self.max_action = max_action

    def generate(self, x):
        if self.tr_noise and self.x_coor >= 2.133: 
            return (0.07*torch.randn_like(x)).clamp(-0.175, 0.175)
        if self.x_coor >= math.pi: 
            return 0.0
        with torch.no_grad():
            eps = self.scale * self.max_action * (math.cos(self.x_coor) + 1.0)
            lim = 2.5*eps
            self.x_coor += 3e-5
            return (eps*torch.randn_like(x)).clamp(-lim, lim)

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
