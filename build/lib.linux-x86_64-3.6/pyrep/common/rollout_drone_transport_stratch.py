from pyrep.policies.maddpg_drone_agent_scratch import Agent
from pyrep.common.replay_buffer import Buffer
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.agents = self._init_agents()
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = os.getcwd()+"/log_drone3_stratch"

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents
    
    def evaluate(self):
        # reset the environment
        s = self.env.reset_world()
        while 1:
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], 0, 0)
                    actions.append(action)
            s_next, r, done = self.env.step(actions)
            s = s_next

