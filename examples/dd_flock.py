#!/usr/bin/env python 
from os.path import dirname, join, abspath
from pyrep import PyRep

# from pyrep.envs.Multi_drones_transportation import Drone_Env
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
# from pyrep.robots.end_effectors.uarm_Vacuum_Gripper import UarmVacuumGripper
import matplotlib
import matplotlib.pyplot as plt
import random
import torch

from collections import OrderedDict
from tensorboardX import SummaryWriter
import os
from collections import deque
import math
from pyrep.backend import sim
import imageio
import argparse

from pyrep.envs.bacterium_environment import Drone_Env
# from pyrep.policies.dqn import DQNAgent
from pyrep.policies.maddpg_agent import Agent
from pyrep.common.replay_buffer import Buffer
from pyrep.common.arguments import get_args
from pyrep.common.rollout_flock import Rollout

if __name__ == '__main__':
    # get the params
    args = get_args()

    env_name = join(dirname(abspath(__file__)), 'RL_flock.ttt')

    num_agents = 3
    # create multiagent environment
    env = Drone_Env(env_name,num_agents)

    args.time_steps = 55*3000 #timesteps*episodes 
    args.max_episode_len = 55
    args.n_agents = num_agents-1 # agent number in a swarm
    args.evaluate_rate = 3000 
    args.evaluate = False #
    args.evaluate_episode_len = 55
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./model_flock"
    args.scenario_name = "flock"
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2)
    args.high_action = 1
    args.low_action = -1

    rollout = Rollout(args, env)
    if args.evaluate:
        returns = rollout.evaluate()
        print('Average returns is', returns)
    else:
        rollout.run()
    
    env.shutdown()








