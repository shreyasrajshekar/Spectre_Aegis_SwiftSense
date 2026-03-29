import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class D3QN_LSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(D3QN_LSTM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # LSTM for Historical Connectivity Analysis
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Dueling Streams: Value and Advantage
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state_seq, hidden_state=None):
        # state_seq: (Batch, Sequence, Features)
        lstm_out, hidden_state = self.lstm(state_seq, hidden_state)
        
        # Get the output from the last time step
        last_out = lstm_out[:, -1, :]
        
        value = self.value_stream(last_out)
        advantage = self.advantage_stream(last_out)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, hidden_state

class RLController:
    def __init__(self, state_dim=3, action_dim=40, sequence_length=10, lr=1e-3, gamma=0.99, device=None):
        # Action space: 40 (10 Freqs * 4 Beams)
        # State: [Occupancy, Power/100, Priority]
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Dynamic AI Configuration Parameters
        self.prediction_window_ms = 100
        self.collision_threshold = 0.8
        self.history_weight = 0.8
        self.uncertainty_weight = 0.2
        self.exp_decay_lambda = 0.1  # Decay rate for temporal weighting
        
        # Auditing Logger
        self.audit_log = deque(maxlen=1000)
        
        self.q_network = D3QN_LSTM(state_dim, action_dim).to(self.device)
        self.target_network = D3QN_LSTM(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        self.last_state = None
        self.last_action = None
        self.loss = 0.0
        
        self.historical_state = deque(maxlen=self.sequence_length)
        # Pad initial state
        for _ in range(self.sequence_length):
            self.historical_state.append([0.0, -1.0, 0.0]) # Idle, -100dB, No Priority

    def push_state(self, is_busy: bool, power_db: float, priority: float):
        """Append normalized telemetry to rolling history"""
        # Power normalized from -100...0 to 0...1
        norm_pwr = (power_db + 100) / 100.0
        self.historical_state.append([1.0 if is_busy else 0.0, norm_pwr, priority])
        
    def get_current_sequence(self):
        """Returns tensor of shape (1, SequenceLength, Features) with temporal exponential decay applied to occupancy"""
        seq = np.array(self.historical_state, dtype=np.float32)
        # Apply exponential decay S_new = S_t * e^(-lambda * steps_ago)
        # where steps_ago is 0 for the most recent state and increases for older states.
        num_steps = len(seq)
        for i in range(num_steps):
            steps_ago = num_steps - 1 - i
            decay_factor = np.exp(-self.exp_decay_lambda * steps_ago)
            seq[i, 0] *= decay_factor  # Apply decay to Occupancy channel
            
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        return tensor

    def select_action(self):
        """Returns (Channel Index, Beam Index) from the 40-dim action space"""
        state_seq = self.get_current_sequence()
        self.last_state = state_seq.clone()
        
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values, _ = self.q_network(state_seq)
                action_idx = q_values.argmax(dim=1).item()
                
        self.last_action = action_idx
        # Map 40-dim flat index to (Freq, Beam)
        ch_idx = action_idx // 4
        beam_idx = action_idx % 4
        return ch_idx, beam_idx

    def push_transition(self, reward):
        """Stores the transition into the replay memory."""
        if self.last_state is not None and self.last_action is not None:
            next_state = self.get_current_sequence()
            self.memory.append((
                self.last_state.cpu().numpy(), 
                self.last_action, 
                reward, 
                next_state.cpu().numpy(), 
                False
            ))

    def update(self):
        """Samples a batch from the replay buffer and trains the D3QN."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.vstack([b[0] for b in batch])).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        
        # Double DQN Logic
        with torch.no_grad():
            next_q_online, _ = self.q_network(next_states)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.target_network(next_states)
            max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        curr_q, _ = self.q_network(states)
        curr_q = curr_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        tau = 0.005
        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
            
        self.loss = loss.item()
        return self.loss

    def get_occupancy_trend(self, n=5):
        """Calculates the trend slope of the last N occupancy states."""
        if len(self.historical_state) < n:
            return 0.0, 'stable'
        # Get last N occupancy states
        y = np.array([state[0] for state in list(self.historical_state)[-n:]])
        x = np.arange(n)
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.05:
            return slope, 'rising'
        elif slope < -0.05:
            return slope, 'falling'
        return slope, 'stable'

    def predict_future_occupancy(self):
        """6G Temporal Twin: Predicts probability of occupancy for multiple horizons and calculates Q-Entropy."""
        state_seq = self.get_current_sequence()
        
        # Base prediction on recent occupancy history
        recent_occupancy = sum([state[0] for state in self.historical_state]) / max(len(self.historical_state), 1)
        
        with torch.no_grad():
            q_values, _ = self.q_network(state_seq)
            # Calculate Probability distribution via Softmax
            probabilities = torch.softmax(q_values, dim=1)
            # Compute Shannon Entropy: H(Q) = -sum(P * log2(P))
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-9)).item()
            # Normalize entropy (max entropy for 40 actions is log2(40) ~ 5.32)
            max_entropy = np.log2(self.action_dim)
            normalized_entropy = min(entropy / max_entropy, 1.0)
            
        slope, trend_str = self.get_occupancy_trend()
        
        # Combine history and model uncertainty
        base_pred = (recent_occupancy * self.history_weight) + (normalized_entropy * self.uncertainty_weight)
        
        # Multi-horizon forecast [100ms, 200ms, 500ms]
        pred_100 = float(np.clip(base_pred, 0.05, 0.95))
        pred_200 = float(np.clip(base_pred + slope * 1.0, 0.05, 0.95)) # Extrapolate trend slightly
        pred_500 = float(np.clip(base_pred + slope * 4.0, 0.05, 0.95)) # Extrapolate trend further
        
        prediction_confidence = 1.0 - normalized_entropy
        
        return prediction_confidence, trend_str, [pred_100, pred_200, pred_500]
        
    def compute_reward(self, is_busy: bool, action_channel: int) -> float:
        """
        Rewards successful throughput vs collision risk.
        If channel is busy (Collision) -> Large Penalty
        If channel is idle (Throughput) -> Reward
        """
        if is_busy:
            return -50.0  # Collision Penalty
        return 10.0       # Successful Transmit Reward

    def step_decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
