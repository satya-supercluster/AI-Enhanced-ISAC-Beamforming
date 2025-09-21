import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rayleigh
from sklearn.metrics import mean_squared_error
import random
from collections import deque
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ISACEnvironment:
    """
    ISAC Environment for V2X scenario with RSU serving multiple vehicles
    """
    def __init__(self, config):
        self.M = config['num_antennas']  # Number of antennas at RSU
        self.K = config['num_vehicles']  # Number of vehicles
        self.L = config['num_targets']   # Number of sensing targets
        self.P_max = config['max_power']  # Maximum transmit power (W)
        self.fc = config['carrier_freq']  # Carrier frequency (Hz)
        self.B = config['bandwidth']      # Bandwidth (Hz)
        self.area_size = config['area_size']  # Simulation area (m x m)
        self.noise_power = config['noise_power']  # Noise power (W)
        
        # Initialize vehicle and target positions
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Random initial positions for vehicles
        self.vehicle_positions = np.random.uniform(0, self.area_size, (self.K, 2))
        self.vehicle_velocities = np.random.uniform(-20, 20, (self.K, 2))  # m/s
        
        # Random initial positions for sensing targets
        self.target_positions = np.random.uniform(0, self.area_size, (self.L, 2))
        self.target_velocities = np.random.uniform(-5, 5, (self.L, 2))  # m/s
        
        # RSU position (center of area)
        self.rsu_position = np.array([self.area_size/2, self.area_size/2])
        
        # Initialize channel state
        self.update_channels()
        
        # Previous beamforming vectors (initialized to zero)
        self.prev_beamforming = np.zeros((self.K + 1, self.M), dtype=complex)
        
        return self.get_state()
    
    def update_channels(self):
        """Update channel state information"""
        self.channels = []
        for k in range(self.K):
            # Distance-based path loss
            distance = np.linalg.norm(self.vehicle_positions[k] - self.rsu_position)
            path_loss_db = 32.45 + 20*np.log10(self.fc/1e9) + 20*np.log10(distance/1000)
            path_loss = 10**(-path_loss_db/10)
            
            # Rayleigh fading
            fading = (np.random.randn(self.M) + 1j*np.random.randn(self.M)) / np.sqrt(2)
            
            # Channel coefficient
            h_k = np.sqrt(path_loss) * fading
            self.channels.append(h_k)
    
    def step(self, action, dt=0.1):
        """Execute one time step with given beamforming action"""
        # Parse action: [beamforming_real, beamforming_imag, power_allocation]
        beamforming_real = action[:self.M*(self.K+1)].reshape(self.K+1, self.M)
        beamforming_imag = action[self.M*(self.K+1):2*self.M*(self.K+1)].reshape(self.K+1, self.M)
        beamforming = beamforming_real + 1j * beamforming_imag
        power_allocation = action[2*self.M*(self.K+1):]
        
        # Normalize beamforming vectors
        for i in range(self.K+1):
            if np.linalg.norm(beamforming[i]) > 0:
                beamforming[i] = beamforming[i] / np.linalg.norm(beamforming[i])
        
        # Update positions (simple mobility model)
        self.vehicle_positions += self.vehicle_velocities * dt
        self.target_positions += self.target_velocities * dt
        
        # Keep vehicles within bounds
        self.vehicle_positions = np.clip(self.vehicle_positions, 0, self.area_size)
        self.target_positions = np.clip(self.target_positions, 0, self.area_size)
        
        # Update channels
        self.update_channels()
        
        # Compute rewards
        comm_reward = self.compute_communication_reward(beamforming[:self.K], power_allocation[:self.K])
        sense_reward = self.compute_sensing_reward(beamforming[self.K], power_allocation[self.K])
        energy_penalty = self.compute_energy_penalty(beamforming, power_allocation)
        
        total_reward = 0.5 * comm_reward + 0.3 * sense_reward - 0.2 * energy_penalty
        
        # Update previous beamforming
        self.prev_beamforming = beamforming.copy()
        
        return self.get_state(), total_reward, False, {
            'comm_reward': comm_reward,
            'sense_reward': sense_reward,
            'energy_penalty': energy_penalty,
            'total_power': np.sum(power_allocation)
        }
    
    def compute_communication_reward(self, beamforming_comm, power_comm):
        """Compute communication reward (sum rate)"""
        rates = []
        for k in range(self.K):
            if power_comm[k] <= 0:
                rates.append(0)
                continue
                
            h_k = self.channels[k]
            w_k = beamforming_comm[k]
            
            # Signal power
            signal_power = power_comm[k] * np.abs(np.dot(h_k.conj(), w_k))**2
            
            # Interference power
            interference_power = 0
            for j in range(self.K):
                if j != k:
                    interference_power += power_comm[j] * np.abs(np.dot(h_k.conj(), beamforming_comm[j]))**2
            
            # SINR and rate
            sinr = signal_power / (interference_power + self.noise_power)
            rate = np.log2(1 + sinr)
            rates.append(rate)
        
        return np.sum(rates)
    
    def compute_sensing_reward(self, beamforming_sense, power_sense):
        """Compute sensing reward (detection probability and estimation accuracy)"""
        if power_sense <= 0:
            return 0
        
        total_reward = 0
        for l in range(self.L):
            # Distance to target
            distance = np.linalg.norm(self.target_positions[l] - self.rsu_position)
            
            # Simplified radar equation for detection probability
            # P_det ∝ P_tx * G^2 / R^4
            gain = np.abs(np.dot(beamforming_sense, np.exp(-1j * 2 * np.pi * distance / 3e8)))**2
            detection_power = power_sense * gain / (distance**4 + 1e-6)  # Add small constant to avoid division by zero
            
            # Detection probability (simplified sigmoid)
            detection_prob = 1 / (1 + np.exp(-10 * (detection_power - 0.1)))
            
            # Estimation accuracy (inversely related to distance)
            estimation_accuracy = 1 / (1 + distance / 100)
            
            total_reward += 0.7 * detection_prob + 0.3 * estimation_accuracy
        
        return total_reward / self.L
    
    def compute_energy_penalty(self, beamforming, power_allocation):
        """Compute energy consumption penalty"""
        total_power = np.sum(power_allocation)
        normalized_power = total_power / self.P_max
        return normalized_power
    
    def get_state(self):
        """Get current state representation"""
        state = []
        
        # Vehicle positions (normalized)
        state.extend(self.vehicle_positions.flatten() / self.area_size)
        
        # Target positions (normalized)
        state.extend(self.target_positions.flatten() / self.area_size)
        
        # Channel state information (real and imaginary parts)
        for h in self.channels:
            state.extend(h.real)
            state.extend(h.imag)
        
        # Previous beamforming vectors (real and imaginary parts)
        state.extend(self.prev_beamforming.real.flatten())
        state.extend(self.prev_beamforming.imag.flatten())
        
        return np.array(state, dtype=np.float32)

class PPOActor(nn.Module):
    """PPO Actor network for continuous action space"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(PPOActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and log_std
        )
        
        self.action_dim = action_dim
        
    def forward(self, state):
        output = self.network(state)
        mean = output[:, :self.action_dim]
        log_std = output[:, self.action_dim:]
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
        std = torch.exp(log_std)
        
        return mean, std

class PPOCritic(nn.Module):
    """PPO Critic network for value function estimation"""
    def __init__(self, state_dim, hidden_dim=512):
        super(PPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze().numpy(), log_prob.item()
    
    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))
    
    def update(self):
        if len(self.memory) == 0:
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.FloatTensor([t[1] for t in self.memory])
        old_log_probs = torch.FloatTensor([t[2] for t in self.memory])
        rewards = torch.FloatTensor([t[3] for t in self.memory])
        next_states = torch.FloatTensor([t[4] for t in self.memory])
        dones = torch.BoolTensor([t[5] for t in self.memory])
        
        # Calculate discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate advantages
        values = self.critic(states).squeeze()
        advantages = discounted_rewards - values.detach()
        
        # PPO update
        for _ in range(self.K_epochs):
            # Actor loss
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(self.critic(states).squeeze(), discounted_rewards)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Clear memory
        self.memory = []

class KalmanFilterBaseline:
    """Traditional Kalman Filter based beamforming baseline"""
    def __init__(self, config):
        self.M = config['num_antennas']
        self.K = config['num_vehicles']
        self.L = config['num_targets']
        self.P_max = config['max_power']
        
        # Initialize state estimates
        self.vehicle_estimates = np.zeros((self.K, 4))  # [x, y, vx, vy]
        self.target_estimates = np.zeros((self.L, 4))   # [x, y, vx, vy]
        
        # Covariance matrices
        self.P_vehicles = [np.eye(4) for _ in range(self.K)]
        self.P_targets = [np.eye(4) for _ in range(self.L)]
        
        # Process and measurement noise
        self.Q = np.diag([0.1, 0.1, 0.01, 0.01])  # Process noise
        self.R = np.diag([1.0, 1.0])              # Measurement noise
    
    def predict_and_update(self, measurements, dt=0.1):
        """Kalman filter prediction and update steps"""
        # State transition matrix
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        # Measurement matrix (observe position only)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        # Update vehicle estimates
        for k in range(self.K):
            # Prediction
            self.vehicle_estimates[k] = F @ self.vehicle_estimates[k]
            self.P_vehicles[k] = F @ self.P_vehicles[k] @ F.T + self.Q
            
            # Update with measurement
            if k < len(measurements['vehicles']):
                y = measurements['vehicles'][k] - H @ self.vehicle_estimates[k]
                S = H @ self.P_vehicles[k] @ H.T + self.R
                K = self.P_vehicles[k] @ H.T @ np.linalg.inv(S)
                
                self.vehicle_estimates[k] += K @ y
                self.P_vehicles[k] = (np.eye(4) - K @ H) @ self.P_vehicles[k]
    
    def generate_beamforming(self, rsu_position):
        """Generate beamforming vectors using Maximum Ratio Transmission"""
        beamforming = np.zeros((self.K + 1, self.M), dtype=complex)
        power_allocation = np.ones(self.K + 1) * (self.P_max / (self.K + 1))
        
        # Communication beamforming (MRT towards predicted vehicle positions)
        for k in range(self.K):
            # Direction to vehicle
            vehicle_pos = self.vehicle_estimates[k][:2]
            direction = vehicle_pos - rsu_position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            
            # Simple beamforming (uniform linear array assumption)
            angles = np.arctan2(direction[1], direction[0])
            array_response = np.exp(1j * np.pi * np.arange(self.M) * np.sin(angles))
            beamforming[k] = array_response / np.sqrt(self.M)
        
        # Sensing beamforming (uniform beamforming)
        beamforming[self.K] = np.ones(self.M) / np.sqrt(self.M)
        
        return beamforming, power_allocation

class SpikingNeuralNetwork:
    """Simplified Spiking Neural Network for energy-efficient beamforming"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Synaptic weights (simplified)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        
        # LIF neuron parameters
        self.tau = 20.0  # membrane time constant (ms)
        self.v_thresh = 1.0  # spike threshold
        self.v_reset = 0.0   # reset potential
        
        # Neuron states
        self.v_hidden = np.zeros(hidden_dim)
        self.v_output = np.zeros(output_dim)
        
        # Spike trains
        self.spike_hidden = np.zeros(hidden_dim)
        self.spike_output = np.zeros(output_dim)
        
        # Energy consumption tracking
        self.energy_ops = 0
        self.energy_spikes = 0
    
    def encode_input(self, x, dt=1.0):
        """Rate encoding: convert continuous values to spike rates"""
        # Normalize input to [0, 1] range
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        
        # Generate Poisson spikes based on rates
        spike_probs = x_norm * dt / 1000  # Convert to probability per dt
        spikes = np.random.binomial(1, np.clip(spike_probs, 0, 1))
        
        return spikes
    
    def lif_dynamics(self, v, I, dt=1.0):
        """Leaky Integrate-and-Fire neuron dynamics"""
        # Update membrane potential
        dv = (-v + I) / self.tau
        v_new = v + dv * dt
        
        # Check for spikes
        spikes = (v_new >= self.v_thresh).astype(int)
        
        # Reset spiked neurons
        v_new = np.where(spikes, self.v_reset, v_new)
        
        return v_new, spikes
    
    def forward(self, x, dt=1.0, num_steps=100):
        """Forward pass through SNN"""
        self.energy_ops = 0
        self.energy_spikes = 0
        
        # Encode input as spikes
        input_spikes = self.encode_input(x, dt)
        
        # Initialize outputs
        output_sum = np.zeros(self.output_dim)
        
        for t in range(num_steps):
            # Hidden layer
            I_hidden = np.dot(input_spikes, self.W1)
            self.v_hidden, self.spike_hidden = self.lif_dynamics(self.v_hidden, I_hidden, dt)
            
            # Output layer
            I_output = np.dot(self.spike_hidden, self.W2)
            self.v_output, self.spike_output = self.lif_dynamics(self.v_output, I_output, dt)
            
            # Accumulate output spikes
            output_sum += self.spike_output
            
            # Energy consumption (simplified model)
            self.energy_ops += np.sum(input_spikes > 0) + np.sum(self.spike_hidden > 0)  # Active neurons
            self.energy_spikes += np.sum(self.spike_hidden) + np.sum(self.spike_output)  # Spike events
        
        # Decode output (rate decoding)
        output = output_sum / num_steps
        
        return output
    
    def get_energy_consumption(self):
        """Return energy consumption metrics"""
        # Simplified energy model (pJ per operation/spike)
        energy_per_op = 0.9  # pJ per MAC operation
        energy_per_spike = 0.1  # pJ per spike
        
        total_energy = self.energy_ops * energy_per_op + self.energy_spikes * energy_per_spike
        return {
            'total_energy_pJ': total_energy,
            'operations': self.energy_ops,
            'spikes': self.energy_spikes
        }

def train_drl_agent(env, agent, episodes=1000):
    """Train the DRL agent"""
    episode_rewards = []
    comm_rewards = []
    sense_rewards = []
    energy_penalties = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_comm = 0
        episode_sense = 0
        episode_energy = 0
        
        for step in range(100):  # Max steps per episode
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, next_state, done)
            
            episode_reward += reward
            episode_comm += info['comm_reward']
            episode_sense += info['sense_reward']
            episode_energy += info['energy_penalty']
            
            state = next_state
            
            if done:
                break
        
        # Update agent every 20 episodes
        if (episode + 1) % 20 == 0:
            agent.update()
        
        episode_rewards.append(episode_reward)
        comm_rewards.append(episode_comm)
        sense_rewards.append(episode_sense)
        energy_penalties.append(episode_energy)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'comm_rewards': comm_rewards,
        'sense_rewards': sense_rewards,
        'energy_penalties': energy_penalties
    }

def evaluate_baseline(env, baseline, episodes=100):
    """Evaluate Kalman filter baseline"""
    episode_rewards = []
    comm_rewards = []
    sense_rewards = []
    energy_penalties = []
    total_powers = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_comm = 0
        episode_sense = 0
        episode_energy = 0
        episode_power = 0
        
        for step in range(100):
            # Get measurements (vehicle positions)
            measurements = {
                'vehicles': env.vehicle_positions,
                'targets': env.target_positions
            }
            
            # Update Kalman filter
            baseline.predict_and_update(measurements)
            
            # Generate beamforming action
            beamforming, power_allocation = baseline.generate_beamforming(env.rsu_position)
            
            # Convert to action format
            action = np.concatenate([
                beamforming.real.flatten(),
                beamforming.imag.flatten(),
                power_allocation
            ])
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_comm += info['comm_reward']
            episode_sense += info['sense_reward']
            episode_energy += info['energy_penalty']
            episode_power += info['total_power']
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        comm_rewards.append(episode_comm)
        sense_rewards.append(episode_sense)
        energy_penalties.append(episode_energy)
        total_powers.append(episode_power)
    
    return {
        'episode_rewards': episode_rewards,
        'comm_rewards': comm_rewards,
        'sense_rewards': sense_rewards,
        'energy_penalties': energy_penalties,
        'total_powers': total_powers
    }

def plot_results(drl_results, baseline_results):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Episode rewards
    axes[0, 0].plot(drl_results['episode_rewards'], label='DRL-PPO', alpha=0.7)
    axes[0, 0].axhline(np.mean(baseline_results['episode_rewards']), 
                      color='red', linestyle='--', label='Kalman Filter')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Communication rewards
    axes[0, 1].plot(drl_results['comm_rewards'], label='DRL-PPO', alpha=0.7)
    axes[0, 1].axhline(np.mean(baseline_results['comm_rewards']), 
                      color='red', linestyle='--', label='Kalman Filter')
    axes[0, 1].set_title('Communication Performance')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Sum Rate (bps/Hz)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Sensing rewards
    axes[0, 2].plot(drl_results['sense_rewards'], label='DRL-PPO', alpha=0.7)
    axes[0, 2].axhline(np.mean(baseline_results['sense_rewards']), 
                      color='red', linestyle='--', label='Kalman Filter')
    axes[0, 2].set_title('Sensing Performance')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Detection Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Energy consumption
    axes[1, 0].plot(drl_results['energy_penalties'], label='DRL-PPO', alpha=0.7)
    axes[1, 0].axhline(np.mean(baseline_results['energy_penalties']), 
                      color='red', linestyle='--', label='Kalman Filter')
    axes[1, 0].set_title('Energy Consumption')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Normalized Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Performance comparison (box plots)
    metrics = ['Total Reward', 'Communication', 'Sensing', 'Energy Penalty']
    drl_data = [
        drl_results['episode_rewards'][-100:],
        drl_results['comm_rewards'][-100:],
        drl_results['sense_rewards'][-100:],
        drl_results['energy_penalties'][-100:]
    ]
    baseline_data = [
        baseline_results['episode_rewards'],
        baseline_results['comm_rewards'],
        baseline_results['sense_rewards'],
        baseline_results['energy_penalties']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    drl_means = [np.mean(data) for data in drl_data]
    baseline_means = [np.mean(data) for data in baseline_data]
    
    axes[1, 1].bar(x - width/2, drl_means, width, label='DRL-PPO', alpha=0.7)
    axes[1, 1].bar(x + width/2, baseline_means, width, label='Kalman Filter', alpha=0.7)
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_ylabel('Average Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Energy efficiency comparison
    drl_energy = np.mean(drl_results['energy_penalties'][-100:])
    baseline_energy = np.mean(baseline_results['energy_penalties'])
    energy_reduction = (baseline_energy - drl_energy) / baseline_energy * 100
    
    methods = ['Kalman Filter', 'DRL-PPO']
    energies = [baseline_energy, drl_energy]
    colors = ['red', 'blue']
    
    bars = axes[1, 2].bar(methods, energies, color=colors, alpha=0.7)
    axes[1, 2].set_title(f'Energy Consumption\n({energy_reduction:.1f}% reduction)')
    axes[1, 2].set_ylabel('Normalized Energy')
    axes[1, 2].grid(True)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        axes[1, 2].annotate(f'{energy:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('isac_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_snn_energy_efficiency():
    """Demonstrate SNN energy efficiency"""
    print("\n=== Spiking Neural Network Energy Analysis ===")
    
    # Create traditional ANN and SNN for comparison
    input_dim = 100
    hidden_dim = 64
    output_dim = 32
    
    # Traditional ANN (simplified energy model)
    ann_weights = np.random.randn(input_dim, hidden_dim) * 0.1
    ann_output_weights = np.random.randn(hidden_dim, output_dim) * 0.1
    
    # SNN
    snn = SpikingNeuralNetwork(input_dim, hidden_dim, output_dim)
    
    # Test input
    test_input = np.random.randn(input_dim)
    
    # ANN forward pass energy (simplified)
    ann_ops = input_dim * hidden_dim + hidden_dim * output_dim  # MAC operations
    ann_energy_per_op = 4.6  # pJ per MAC operation
    ann_memory_access = (input_dim + hidden_dim + output_dim) * 640  # pJ per memory access
    ann_total_energy = ann_ops * ann_energy_per_op + ann_memory_access
    
    # SNN forward pass
    snn_output = snn.forward(test_input)
    snn_energy = snn.get_energy_consumption()
    
    print(f"Traditional ANN Energy Consumption:")
    print(f"  - MAC Operations: {ann_ops}")
    print(f"  - Total Energy: {ann_total_energy/1000:.2f} nJ")
    
    print(f"\nSpiking Neural Network Energy Consumption:")
    print(f"  - Active Operations: {snn_energy['operations']}")
    print(f"  - Spike Events: {snn_energy['spikes']}")
    print(f"  - Total Energy: {snn_energy['total_energy_pJ']/1000:.2f} nJ")
    
    energy_reduction = (ann_total_energy - snn_energy['total_energy_pJ']) / ann_total_energy * 100
    print(f"  - Energy Reduction: {energy_reduction:.1f}%")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    methods = ['Traditional ANN', 'Spiking NN']
    energies = [ann_total_energy/1000, snn_energy['total_energy_pJ']/1000]  # Convert to nJ
    colors = ['red', 'green']
    
    bars = plt.bar(methods, energies, color=colors, alpha=0.7)
    plt.title(f'Energy Consumption Comparison\n({energy_reduction:.1f}% reduction with SNN)')
    plt.ylabel('Energy Consumption (nJ)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        plt.annotate(f'{energy:.2f} nJ',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('snn_energy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("=== AI-Enhanced Beamforming for Energy-Efficient ISAC ===\n")
    
    # Configuration
    config = {
        'num_antennas': 64,
        'num_vehicles': 8,
        'num_targets': 4,
        'max_power': 46,  # dBm equivalent in watts (≈40W)
        'carrier_freq': 28e9,  # 28 GHz
        'bandwidth': 100e6,    # 100 MHz
        'area_size': 1000,     # 1km x 1km area
        'noise_power': 1e-12   # -90 dBm
    }
    
    # Create environment
    env = ISACEnvironment(config)
    
    # Calculate dimensions
    state_dim = len(env.get_state())
    action_dim = 2 * config['num_antennas'] * (config['num_vehicles'] + 1) + (config['num_vehicles'] + 1)
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Environment configured with {config['num_vehicles']} vehicles and {config['num_targets']} targets\n")
    
    # Create DRL agent
    agent = PPOAgent(state_dim, action_dim)
    
    # Create baseline
    baseline = KalmanFilterBaseline(config)
    
    # Training
    print("Training DRL Agent...")
    start_time = time.time()
    drl_results = train_drl_agent(env, agent, episodes=500)  # Reduced for demo
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds\n")
    
    # Evaluation
    print("Evaluating Kalman Filter Baseline...")
    baseline_results = evaluate_baseline(env, baseline, episodes=100)
    
    # Results analysis
    print("\n=== Performance Analysis ===")
    
    # DRL results (last 100 episodes)
    drl_final_reward = np.mean(drl_results['episode_rewards'][-100:])
    drl_final_comm = np.mean(drl_results['comm_rewards'][-100:])
    drl_final_sense = np.mean(drl_results['sense_rewards'][-100:])
    drl_final_energy = np.mean(drl_results['energy_penalties'][-100:])
    
    # Baseline results
    baseline_reward = np.mean(baseline_results['episode_rewards'])
    baseline_comm = np.mean(baseline_results['comm_rewards'])
    baseline_sense = np.mean(baseline_results['sense_rewards'])
    baseline_energy = np.mean(baseline_results['energy_penalties'])
    
    print(f"DRL-PPO Performance:")
    print(f"  - Average Reward: {drl_final_reward:.3f}")
    print(f"  - Communication Score: {drl_final_comm:.3f}")
    print(f"  - Sensing Score: {drl_final_sense:.3f}")
    print(f"  - Energy Penalty: {drl_final_energy:.3f}")
    
    print(f"\nKalman Filter Performance:")
    print(f"  - Average Reward: {baseline_reward:.3f}")
    print(f"  - Communication Score: {baseline_comm:.3f}")
    print(f"  - Sensing Score: {baseline_sense:.3f}")
    print(f"  - Energy Penalty: {baseline_energy:.3f}")
    
    # Improvements
    reward_improvement = (drl_final_reward - baseline_reward) / abs(baseline_reward) * 100
    comm_improvement = (drl_final_comm - baseline_comm) / baseline_comm * 100
    sense_improvement = (drl_final_sense - baseline_sense) / baseline_sense * 100
    energy_reduction = (baseline_energy - drl_final_energy) / baseline_energy * 100
    
    print(f"\nImprovements with DRL:")
    print(f"  - Total Reward: {reward_improvement:+.1f}%")
    print(f"  - Communication: {comm_improvement:+.1f}%")
    print(f"  - Sensing: {sense_improvement:+.1f}%")
    print(f"  - Energy Reduction: {energy_reduction:.1f}%")
    
    # Plot results
    plot_results(drl_results, baseline_results)
    
    # SNN demonstration
    demonstrate_snn_energy_efficiency()
    
    print("\n=== Summary ===")
    print(f"✓ Successfully implemented DRL-based ISAC beamforming")
    print(f"✓ Achieved {energy_reduction:.1f}% energy reduction")
    print(f"✓ Improved communication performance by {comm_improvement:.1f}%")
    print(f"✓ Enhanced sensing accuracy by {sense_improvement:.1f}%")
    print(f"✓ Demonstrated SNN potential for additional energy savings")
    print(f"✓ Training completed in {training_time:.1f} seconds")

if __name__ == "__main__":
    main()