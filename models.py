import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MEMORY_SIZE = 500000
BATCH_SIZE = 512
GAMMA = 0.96
TAU = 0.01
POLICY_NOISE = 0.3
NOISE_CLIP = 0.5
POLICY_FREQ = 2
EXPLORATION_NOISE = 0.4

# Res Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x):
        return x + self.net(x)

# Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.PReLU()
        )
        
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(0.1)
        self.max_action = max_action
        self.to(device)
    
    def forward(self, state):
        x = state.to(device)
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return self.max_action * x
        

# Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.PReLU()
        )

        # Q1
        self.q1_res1 = ResidualBlock(256)
        self.q1_res2 = ResidualBlock(256)
        self.q1_output = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

        # Q2
        self.q2_res1 = ResidualBlock(256)
        self.q2_res2 = ResidualBlock(256)
        self.q2_output = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

        self.to(device)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = self.input_layer(sa)

        # Q1
        q1 = self.q1_res1(x)
        q1 = self.q1_res2(q1)
        q1 = self.q1_output(q1)

        # Q2
        q2 = self.q2_res1(x)
        q2 = self.q2_res2(q2)
        q2 = self.q2_output(q2)

        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        x = self.input_layer(sa)
        x = self.q1_res1(x)
        x = self.q1_res2(x)
        q1 = self.q1_output(x)

        return q1

# ReplayBuffer
class ReplayBuffer:
    def __init__(self, max_size=MEMORY_SIZE):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.alpha = 0.5
        self.beta = 0.6
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        # sample probility
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        # sampling weight
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).to(device),
            torch.FloatTensor(weights).to(device),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)

# TD3
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.total_it = 0
        self.scaler = GradScaler(device=device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=BATCH_SIZE):
        self.total_it += 1

        state, action, reward, next_state, done, weights, indices = self.replay_buffer.sample(batch_size)

        with autocast(device_type="cuda", dtype=torch.float16):
            # select action and add noise
            noise = torch.FloatTensor(np.random.normal(0, POLICY_NOISE, size=(batch_size, action.size(-1)))).to(device)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * GAMMA * target_Q.detach()

            current_Q1, current_Q2 = self.critic(state, action)

            # TD errors
            td_error1 = (target_Q - current_Q1).abs()
            td_error2 = (target_Q - current_Q2).abs()
            td_errors = torch.max(td_error1, td_error2).squeeze().detach().cpu().numpy()

            critic_loss = (weights * (F.smooth_l1_loss(current_Q1, target_Q, reduction='none') + 
                                F.smooth_l1_loss(current_Q2, target_Q, reduction='none'))).mean()

        # optimize critic
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()

        # update priority
        self.replay_buffer.update_priorities(indices, td_errors)

        # dalayed upodate
        if self.total_it % POLICY_FREQ == 0:
            with autocast(device_type="cuda", dtype=torch.float16):
                actor_loss = -(weights * self.critic.Q1(state, self.actor(state))).mean()

            # optimize actor
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()

            # update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")
        self.critic.to(device)
        self.actor.to(device)

    def load(self, filename):
        try:
            # load actor
            actor_state_dict = torch.load(filename + "_actor", map_location=device)
            self.actor.load_state_dict(actor_state_dict)
            
            # load critic
            critic_state_dict = torch.load(filename + "_critic", map_location=device)
            self.critic.load_state_dict(critic_state_dict)
            
            # sync target networks
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Model loaded failure: {e}")