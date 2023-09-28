# main code that contains the neural network setup and policy + critic updates
# see ddpg.py for other details in the network

import torch
from pyrep.policies.utilities import soft_update, transpose_to_tensor, transpose_list, hard_update
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# individual network settings for each actor + critic pair
from pyrep.policies.network import Actor, Critic
from pyrep.policies.utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np

from pyrep.policies.noise import OUNoise, BetaNoise, GaussNoise, WeightedNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    def __init__(self, action_size, action_type, state_size, hidden_in_size, hidden_out_size, num_atoms, lr_actor, lr_critic, l2_decay, noise_type, OU_mu, OU_theta, OU_sigma):
        super(DDPGAgent, self).__init__()

        # creating actors, critics and targets using the specified layer sizes. Note for the critics we assume 6 agents
        self.actor = Actor(action_size, state_size, hidden_in_size, hidden_out_size, action_type).to(device)

        self.critic = Critic(action_size, state_size, hidden_in_size, hidden_out_size, num_atoms).to(device)
        self.target_actor =  Actor(action_size, state_size, hidden_in_size, hidden_out_size, action_type).to(device)
        self.target_critic = Critic(action_size, state_size, hidden_in_size, hidden_out_size, num_atoms).to(device)
        self.noise_type = noise_type
        self.action_type = action_type
        
        if noise_type=='OUNoise': # if we're using OUNoise it needs to be initialised as it is an autocorrelated process
            self.noise = OUNoise(action_size, OU_mu, OU_theta, OU_sigma)
            
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

       # initialize optimisers using specigied learning rates
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor, weight_decay=l2_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=l2_decay)

    def act(self, obs, noise_scale=0.0):
        obs = obs.to(device)
        action = self.actor(obs)
        
        if noise_scale==0.0: # if no noise then just return action as is
            pass
        elif self.noise_type=='OUNoise':
            noise = 1.5*self.noise.noise()
            action += noise_scale*(noise-action)
            action = torch.clamp(action,-1,1)
        elif self.noise_type=='BetaNoise':
            action = BetaNoise(action, noise_scale)
        elif self.noise_type=='GaussNoise':
            action = GaussNoise(action, noise_scale)
        elif self.noise_type=='WeightedNoise':
            action = WeightedNoise(action, noise_scale, self.action_type)
        return action
    
    # target actor is only used for updates not exploration and so has no noise
    def target_act(self, obs):
        obs = obs.to(device)
        action = self.target_actor(obs)
        return action

class SDDPG:
    def __init__(self, p):

        super(SDDPG, self).__init__()
        
        self.Sddpg_agent = DDPGAgent(p['action_size'], p['action_type'], p['state_size'], p['hidden_in_size'], p['hidden_out_size'], p['num_atoms'], p['lr_actor'], p['lr_critic'],p['l2_decay'],p['noise_type'],p['OU_mu'],p['OU_theta'],p['OU_sigma'])
        
        
        self.discount_rate = p['discount_rate']
        self.tau = p['tau']
        self.n_steps = p['n_steps']
        self.num_atoms = p['num_atoms']
        self.vmin = p['vmin']
        self.vmax = p['vmax']
        #self.action_size = p['action_size']
        #self.state_size = p['state_size']
        #self.lr_actor = p['lr_actor']
        #self.lr_critic = p['lr_critic']
        #self.hidden_in_size = p['hidden_in_size']
        #self.hidden_out_size = p['hidden_out_size']
        self.iter = 0
        
        self.atoms = torch.linspace(self.vmin,self.vmax,self.num_atoms).to(device)
        self.atoms = self.atoms.unsqueeze(0)

    def get_actor(self):
        """get actors of all the agents in the MADDPG object"""
        actor = self.Sddpg_agent.actor 
        return actor

    def get_target_actor(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actor = self.Sddpg_agent.target_actor 
        return target_actor

    def act(self, obs_agent, noise_scale=np.zeros(2)):
        """get actions from all agents in the MADDPG object"""
        action = self.Sddpg_agent.act(obs_agent, noise_scale[0]) 
        return action

    def target_act(self, obs_agent):
        """get target network actions from all the agents in the MADDPG object """
        target_action = self.Sddpg_agent.target_act(obs_agent) 
        return target_action

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.Sddpg_agent
        soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
        soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
        
    def hard_update_targets(self,agent_num):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.Sddpg_agent
        hard_update(ddpg_agent.target_actor, ddpg_agent.actor)
        hard_update(ddpg_agent.target_critic, ddpg_agent.critic)
        
    # def initialise_networks(self, agent1_num, agent1_path):
        
    #     checkpoints = [torch.load(agent0_path)[agent0_num],torch.load(agent1_path)[agent1_num]] # load the torch data
        
    #     for agent, checkpoint in zip(self.maddpg_agent,checkpoints):
        
    #         agent.actor.load_state_dict(checkpoint['actor_params'])                          # actor parameters
    #         agent.critic.load_state_dict(checkpoint['critic_params'])                        # critic parameters
    #         agent.actor_optimizer.load_state_dict(checkpoint['actor_optim_params'])          # actor optimiser state
    #         agent.critic_optimizer.load_state_dict(checkpoint['critic_optim_params'])        # critic optimiser state
            
    #         hard_update(agent.target_actor, agent.actor)                                     # hard updates to initialise the targets
    #         hard_update(agent.target_critic, agent.critic)                                   # hard updates to initialise the targets
            

    def update(self, samples, logger):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """
        """update the critics and actors of all the agents """

        # need to transpose samples and convert them to trsnors
        obs, action, reward, next_obs, done = map(transpose_to_tensor1, samples) # get data

        # full versions of obs and actions are needed for the critics
        # obs_full = torch.cat(obs, 1)
        # next_obs_full = torch.cat(next_obs, 1)
        # action_full = torch.cat(action, 1)
        
        agent = self.Sddpg_agent # set the agent being updated
        agent.critic_optimizer.zero_grad() # zero_grad is required before backprop

        target_actions = self.target_act(next_obs) # produces target actions using both critics for the next state, i.e. a(t+1)
        # target_action = torch.cat(target_action, dim=1) # concatinates them together
        
        # Calculate target distribution to update the critic
        target_prob = agent.target_critic(next_obs, target_actions).detach() # target critic probs
        target_dist = self.to_categorical(reward.unsqueeze(-1), target_prob, done.unsqueeze(-1))
        
        # Calculate log probability distribution
        log_prob = agent.critic(obs, action, log=True) # for cross entropy we need log of critic output
        
        # Calculate the critic loss manually using cross entropy
        critic_loss = -(target_dist * log_prob).sum(-1).mean()
        
        # Update critic using gradient descent
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1) # clipping to prevent too big updates
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # get actions of all agents
        # detach the other agents to save computation when computing derivative
        actor_actions = self.Sddpg_agent.actor(obs)
                   
                
        # combine all the actions and observations for input to critic
        # actor_actions = torch.cat(actor_actions, dim=1)

        critic_probs = agent.critic(obs, actor_actions)
        
        # Multiply value probs by atom values and sum across columns to get Q-Value
        expected_reward = (critic_probs * self.atoms).sum(-1)
        # Calculate the actor network loss
        actor_loss = -expected_reward.mean()
        actor_loss.backward(retain_graph=True)
        agent.actor_optimizer.step()
        
        # adding the losses to tensorboard
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses',{'critic loss': cl,'actor_loss': al},self.iter)
       
        
    def to_categorical(self, rewards, probs, dones):
        """
        Credit to Matt Doll and Shangtong Zhang for this function:
        https://github.com/whiterabbitobj
        https://github.com/ShangtongZhang
        """

        # Create local vars to keep code more concise
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atomslogger
        n_steps = self.n_steps
        discount_rate = self.discount_rate
        
        # this is the increment between atoms
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # projecting the rewards to the atoms
        projected_atoms = rewards + discount_rate**n_steps * atoms * (1 - dones)
        projected_atoms.clamp_(vmin, vmax) # vmin/vmax are arbitary so any observations falling outside this range will be cliped
        b = (projected_atoms - vmin) / delta_z

        # precision is reduced to prevent e.g 99.00000...ceil() evaluating to 100 instead of 99.
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        # initialising projected_probs
        projected_probs = torch.tensor(np.zeros(probs.size())).to(device)

        # a bit like one-hot encoding but for the specified atoms
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()
