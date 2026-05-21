import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from agents.replay_buffer import ReplayBuffer


# =========================================================
# Q NETWORK
#
# Neural approximation of:
#
# Q(s, a)
#
# Still uses DISCRETE ACTION SPACE
# =========================================================

class QNetwork(nn.Module):

    def __init__(
        self,
        observation_dim,
        action_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(observation_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim)
        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# DQN AGENT
#
# Compatible with:
# - Gymnasium
# - Dict observation (after flattening)
# - Box observation
# - Discrete action space
# =========================================================

class DQNAgent:

    def __init__(
        self,
        observation_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):

        self.observation_dim = observation_dim

        self.action_dim = action_dim

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.policy_net = QNetwork(
            self.observation_dim,
            self.action_dim
        ).to(self.device)

        self.target_net = QNetwork(
            self.observation_dim,
            self.action_dim
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr
        )

        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer()

    # =====================================================
    # ACTION SELECTION
    #
    # epsilon-greedy exploration
    # =====================================================

    def predict(self, observation):

        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.action_dim - 1
            )

        obs_tensor = torch.FloatTensor(
            observation
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.policy_net(obs_tensor)

        return torch.argmax(q_values).item()

    # =====================================================
    # TRAIN STEP
    # =====================================================

    def train_step(self, batch_size=64):

        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )

        states = torch.FloatTensor(
            states
        ).to(self.device)

        actions = torch.LongTensor(
            actions
        ).unsqueeze(1).to(self.device)

        rewards = torch.FloatTensor(
            rewards
        ).unsqueeze(1).to(self.device)

        next_states = torch.FloatTensor(
            next_states
        ).to(self.device)

        dones = torch.FloatTensor(
            dones
        ).unsqueeze(1).to(self.device)

        # =================================================
        # Current Q-values
        # =================================================

        current_q = self.policy_net(states).gather(
            1,
            actions
        )

        # =================================================
        # Target Q-values
        #
        # y = r + gamma * max(Q_target)
        # =================================================

        with torch.no_grad():

            max_next_q = self.target_net(
                next_states
            ).max(1)[0].unsqueeze(1)

        target_q = rewards + (
            self.gamma *
            max_next_q *
            (1 - dones)
        )

        # =================================================
        # LOSS
        # =================================================

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # # =================================================
        # # Epsilon decay
        # # =================================================

        # self.epsilon = max(
        #     self.epsilon_min,
        #     self.epsilon * self.epsilon_decay
        # )

    # =====================================================
    # TARGET NETWORK UPDATE
    # =====================================================

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # =====================================================
    # SAVE / LOAD
    # =====================================================

    def save(self, path):

        torch.save(
            self.policy_net.state_dict(),
            path
        )

    def load(self, path):

        self.policy_net.load_state_dict(
            torch.load(path)
        )

        self.policy_net.eval()


    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )