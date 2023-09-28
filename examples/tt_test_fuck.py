from os.path import dirname, join, abspath
import numpy as np
from pyrep.envs.drone_transport_env_fuck import Drone_Env_transport_TEST
from scipy.io import savemat
from mpi4py import MPI
from pyrep.common.arguments import get_args
import os 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# get the params
args = get_args()

SCENE_FILE = join(dirname(abspath(__file__)), 'fuck_load.ttt')
num_agents = 1
# create multiagent environment

env = Drone_Env_transport_TEST(args, rank, SCENE_FILE, num_agents)

# signal.pause()
while 1:
    env.step()  







