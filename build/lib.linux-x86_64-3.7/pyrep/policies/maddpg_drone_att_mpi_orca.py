import torch
import os
from pyrep.networks.actor_critic import Actor1, Critic1
import numpy as np

 # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor1(args,agent_id)
        self.critic_network = Critic1(args)

        # build up the target network
        self.actor_target_network = Actor1(args,agent_id)
        self.critic_target_network = Critic1(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir) 
        
        self.model_path = self.args.save_dir + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.args.evaluate:
            if os.path.exists(self.model_path + '/params.pkl'):
                self.initialise_networks(self.model_path+'/params.pkl')
                print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/params.pkl'))
        # self.pol_dev = 'cpu'  # device for policies
        # self.critic_dev = 'cpu'  # device for critics
        # self.trgt_pol_dev = 'cpu'  # device for target policies
        # self.trgt_critic_dev = 'cpu'  # device for target critics

        self.niter = 0
                                                                    
    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    def select_action(self, o, noise_rate, epsilon, use_gpu,logger):
        #[none, sdim]
        us = []
        for i in range(o.shape[0]):
            if np.random.uniform() < epsilon:
                u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
            else:
                inputs = torch.tensor(o[i,:], dtype=torch.float32).unsqueeze(0)
                pi = self.actor_network(inputs).squeeze(0)
                # print('{} : {}'.format(self.name, pi))
                u = pi.cpu().numpy()
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
                # print("noise: ", np.random.randn(*u.shape))
                u += noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
            us.append(u)
        return np.array(us).copy()

    # update the network
    def train(self, transitions, other_agents, logger, use_gpu):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward [none,]
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].actor_target_network(o_next[agent_id])) #n*[none, adim]
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach() #[none,1]

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u, logger=logger, iter_n=self.niter) 
        critic_loss = (target_q - q_value).pow(2).mean()

        if logger is not None:
            logger.add_scalar('Agent%d/losses/q_loss'%self.agent_id, critic_loss, self.niter)

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if logger is not None:
            logger.add_scalar('Agent%d/losses/a_loss'%self.agent_id, actor_loss, self.niter)

        self.niter += 1
        # print(self.niter)

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def prep_training(self, device='gpu'):
        self.critic_network.train()
        self.critic_target_network.train()
        self.actor_network.train()
        self.actor_target_network.train()

        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()

        # if not self.pol_dev == device:
        #     self.actor_network = fn(self.actor_network)
        #     self.pol_dev = device
        # if not self.critic_dev == device:
        #     self.critic_network = fn(self.critic_network)
        #     self.critic_dev = device
        # if not self.trgt_pol_dev == device:
        #     self.actor_target_network = fn(self.actor_target_network)
        #     self.trgt_pol_dev = device
        # if not self.trgt_critic_dev == device:
        #     self.critic_target_network = fn(self.critic_target_network)
        #     self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        
        self.actor_network.eval()
        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()
        # # only need main policy for rollouts
        # if not self.pol_dev == device:
        #     self.actor_network = fn(self.actor_network)
        #     self.pol_dev = device

    #--------------------ORCA--------------------------------
    def ORCA_distance(self, vel, points):
        #input is array, vel is tensor
        point = vel
        line_point1 = points[:2]
        line_point2 = points[:2]+points[2:4]

        vec1 = line_point1-point
        vec2 = line_point2-point

        vec1_3d = torch.zeros(3)
        vec2_3d = torch.zeros(3)
        vec1_3d[0] = vec1[0]
        vec1_3d[1] = vec1[1]
        vec2_3d[0] = vec2[0]
        vec2_3d[1] = vec2[1]

        distance = torch.abs(torch.cross(vec1_3d,vec2_3d)[2])/torch.linalg.norm(line_point1-line_point2)

        return distance

    # update the network
    def train_ORCA(self, transitions, other_agents,logger, use_gpu):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        p_vel = transitions['prefer_vel_%d' % self.agent_id]
        plane = transitions['plane_%d' % self.agent_id] #[none,n_agent,plane_d]
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        if logger is not None:
            logger.add_scalar('Agent%d/losses/q_orca_loss'%self.agent_id, critic_loss, self.niter)

        # the actor loss
        # use closest distance to half plane to optimize actor
        # self.actor_network(o[self.agent_id])
        loss = torch.zeros(plane.shape[0])
        
        v1 = self.actor_network(o[self.agent_id]) #[none,2]
        d_sum_record = 0
        v_sum_record = 0
        for i in range(plane.shape[0]):
            d_sum = 0
            for j in range(plane[i,36].to(torch.int)):
                d_sum += self.ORCA_distance(v1[i,:],plane[i,j*4:j*4+4])               
            loss_local = d_sum + (v1[i,:]-p_vel[i,:]).pow(2).sum()
            loss[i] = loss_local
            d_sum_record += d_sum
            v_sum_record += (v1[i,:]-p_vel[i,:]).pow(2).sum()

        if logger is not None:
            logger.add_scalar('Agent%d/losses/actor_orca_distance'%self.agent_id, d_sum_record, self.niter)
            logger.add_scalar('Agent%d/losses/actor_orca_velocity_loss'%self.agent_id, v_sum_record, self.niter)
        
        actor_loss = loss.mean()
        # if self.agent_id == 0:
        # print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if logger is not None:
            logger.add_scalar('Agent%d/losses/a_orca_loss'%self.agent_id, actor_loss, self.niter)

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
        self.niter += 1


    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.actor_network.load_state_dict(checkpoint['actor_params'])    # actor parameters
        self.critic_network.load_state_dict(checkpoint['critic_params'])    # critic parameters
        self.actor_optim.load_state_dict(checkpoint['actor_optim_params']) # actor optimiser state
        self.critic_optim.load_state_dict(checkpoint['critic_optim_params']) # critic optimiser state

        self.actor_target_network.load_state_dict(checkpoint['actor_target_params']) 
        self.critic_target_network.load_state_dict(checkpoint['critic_target_params'])
        
    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'actor_params' : self.actor_network.state_dict(),
                    'actor_optim_params': self.actor_optim.state_dict(),
                    'actor_target_params': self.actor_target_network.state_dict(),
                    'critic_params' : self.critic_network.state_dict(),
                    'critic_target_params' : self.critic_target_network.state_dict(),
                    'critic_optim_params' : self.critic_optim.state_dict()}

        torch.save(save_dict, self.model_path + '/' + num + '_params.pkl')



