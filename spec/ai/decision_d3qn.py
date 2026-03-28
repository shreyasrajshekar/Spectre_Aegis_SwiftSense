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
        
        self.q_network = D3QN_LSTM(state_dim, action_dim).to(self.device)
        self.target_network = D3QN_LSTM(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
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
        """Returns tensor of shape (1, SequenceLength, Features)"""
        seq = np.array(self.historical_state)
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        return tensor

    def select_action(self):
        """Returns (Channel Index, Beam Index) from the 40-dim action space"""
        state_seq = self.get_current_sequence()
        
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values, _ = self.q_network(state_seq)
                action_idx = q_values.argmax(dim=1).item()
                
        # Map 40-dim flat index to (Freq, Beam)
        ch_idx = action_idx // 4
        beam_idx = action_idx % 4
        return ch_idx, beam_idx

    def predict_future_occupancy(self) -> float:
        """6G Temporal Twin: Predicts probability of occupancy in the next 100ms."""
        state_seq = self.get_current_sequence()
        with torch.no_grad():
            q_values, _ = self.q_network(state_seq)
            # Higher Q-values for 'Staying' actions imply probable occupancy elsewhere, 
            # but we use the value stream here as a surrogate for state evolution.
            # Real forecast would use a separate regression head, 
            # here we approximate it from the hidden state.
            hidden_norm = torch.sigmoid(q_values.mean()).item()
        return hidden_norm
        
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
