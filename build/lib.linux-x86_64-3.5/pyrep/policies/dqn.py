import numpy as np
import torch
import torch.optim as optim
from pyrep.policies.network1 import DQN,DQN_conv
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pyrep.policies.buffer import ReplayBuffer
from pyrep.policies.agent_experience_buffer import AgentReplayMemory
from pyrep.policies.utilities1 import soft_update, transpose_to_tensor, transpose_list, hard_update
import torch.nn.functional as f
from collections import namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = 'cpu'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        n_agents,
        buffer_size: int = 1e5,
        batch_size: int = 256,
        gamma: float = 0.995,
        tau: float = 1e-3,
        learning_rate: float = 7e-4,
        update_every: int = 4,
    ):
        """
        Initialize DQN agent using the agent-experience buffer

        Args:
            state_size (int): Size of the state observation returned by the
                environment
            action_size (int): Action space size
            n_agents (int): Number of agents in the environment
            buffer_size (int): Desired total experience buffer size
            batch_size (int): Mini-batch size
            gamma (float): Discount factor
            tau (float): For soft update of target parameters
            learning_rate (float): Learning rate
            update_every (int): Number of steps before target network update
        """

        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # Q-Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.policy_net2 = DQN_conv(40, 90, action_size).to(device)
        self.target_net2 = DQN_conv(40, 90, action_size).to(device)
        self.target_net2.load_state_dict(self.policy_net2.state_dict())

        self.optimizer2 = optim.RMSprop(self.policy_net2.parameters())
        self.memory2 = ReplayMemory(10000)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(n_agents,buffer_size, n_steps = 5, discount_rate = gamma)
        self.memory1 = AgentReplayMemory(buffer_size, n_agents, state_size, device)
        self.buffer_size = buffer_size
        self.t_step = 0

        self.update_every = update_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau


    def step2(self, states, actions, rewards, next_steps, done):

        # self.memory2.push_agent_actions(states, actions, rewards, next_steps, done)
        # Store the transition in memory
        self.memory2.push(states, actions, next_steps, rewards)

        if len(self.memory2) > self.batch_size:
            experience = self.memory2.sample(self.batch_size)
            self.learn2(experience, self.gamma)

    def step1(self, states, actions, rewards, next_steps, done):

        self.memory1.push_agent_actions(states, actions, rewards, next_steps, done)

        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0:
            if self.memory1.at_capacity():
                experience = self.memory1.sample(self.batch_size)
                loss = self.learn1(experience, self.gamma)
                # print("wawa")
                return loss
        return 0

    def step(self, states, actions, rewards, next_steps, dones):
        transition = ([states, actions, rewards, next_steps, dones])
        self.memory.push(transition)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory.deque) == self.buffer_size:
                experience = self.memory.sample(self.batch_size)
                self.learn(experience, self.gamma)

    def act2(self, states, eps=0):
        # print(states.shape)
        # states = torch.from_numpy(states).float().to(device)
        self.policy_net2.eval()

        with torch.no_grad():
            action_values = self.policy_net2(states)
            
        self.policy_net2.train()

        r = np.random.random(size=self.n_agents)

        action_values = np.argmax(action_values.cpu().data.numpy(), axis=1)
        random_choices = np.random.randint(0, self.action_size, size=self.n_agents)
        
        return np.where(r > eps, action_values, random_choices)
        

    def act(self, states, eps=0):
        states = torch.from_numpy(states).float().to(device)
        self.policy_net.eval()

        with torch.no_grad():
            action_values = self.policy_net(states)
        self.policy_net.train()

        r = np.random.random(size=self.n_agents)

        action_values = np.argmax(action_values.cpu().data.numpy(), axis=1)
        random_choices = np.random.randint(0, self.action_size, size=self.n_agents)

        return np.where(r > eps, action_values, random_choices)

    def learn2(self, experiences, gamma):
        batch = Transition(*zip(*experiences))
        #Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # print(batch.action)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        self.policy_net2.train()
        self.target_net2.eval()
        # print(action_batch.shape)
        # shape of output from the model (batch_size,action_dim) = (64,4)
        # print(self.policy_net2(state_batch).shape)
        predicted_targets = self.policy_net2(state_batch).gather(1, action_batch)

        labels_next = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            labels_next[non_final_mask] = self.target_net2(non_final_next_states).max(1)[0].detach()

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = (labels_next * gamma) + reward_batch

        loss = f.smooth_l1_loss(predicted_targets, labels.unsqueeze(1))
        
        self.optimizer2.zero_grad()
        loss.backward()

        for param in self.policy_net2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer2.step()

        # # ------------------- update target network ------------------- #
        # self.soft_update(self.policy_net2, self.target_net2, self.tau)

    def learn1(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        criterion = torch.nn.MSELoss()
        self.policy_net.train()
        self.target_net.eval()

        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_net, self.target_net, self.tau)
        return loss.item()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = map(transpose_to_tensor, experiences)
        
        # states, actions, rewards, next_states, dones = experiences dimension: agent x batch x observation
        states = torch.cat(states,dim =0).to(device)
        actions = torch.cat(actions,dim =0)[
            :, np.newaxis].long().to(device)
        next_states = torch.cat(next_states, dim=0).to(device)
        rewards = torch.cat(rewards, dim=0)[
            :, np.newaxis].to(device)
        dones = torch.cat(dones, dim=0)[
            :, np.newaxis].to(device)
        # print(states.shape)

        criterion = torch.nn.MSELoss()
        self.policy_net.train()
        self.target_net.eval()

        # shape of output from the model (batch_size,action_dim) = (64,4)
        # print(states.shape)
        predicted_targets = self.policy_net(states).gather(1, actions)
        # print(predicted_targets.shape)
        with torch.no_grad():
            labels_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1).to(device)
            # print(labels_next.shape)
        # print(rewards.shape,dones.shape)
        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        # print(labels.shape)
        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )
