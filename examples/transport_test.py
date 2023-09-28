from os.path import dirname, join, abspath
import numpy as np
from pyrep.envs.drone_transport_env import Drone_Env_transport_TEST
from scipy.io import savemat
from mpi4py import MPI
from pyrep.common.arguments import get_args
import os 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# get the params
args = get_args()

SCENE_FILE = join(dirname(abspath(__file__)), 'RL_bound_transport.ttt')
num_agents = 20
# create multiagent environment

env = Drone_Env_transport_TEST(args, rank, SCENE_FILE, num_agents)
save_path = './'+ 'transport_analysis/{}/'.format(int(args.kp))+args.scenario_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

# signal.pause()
while 1:
    if env.done:
        # energy_swarm = comm.gather(env.energy_swarm, root=0) 
        energy_iter = comm.gather(env.energy_iter, root=0) #env_size,none,20
        force_analysis = comm.gather(env.force_analysis, root=0)
        payload_vy = comm.gather(env.payload_vy, root=0)
        payload_vz = comm.gather(env.payload_vz, root=0)

        if rank == 0:
            print("start saving data")
            # print("start saving data")

            # savemat(save_path+'/storeEnerge_total.mat', mdict={'arr': energy_swarm})
            savemat(save_path+'/storeEnerge_iter.mat', mdict={'arr': energy_iter})
            savemat(save_path+'/storeForce.mat', mdict={'arr': force_analysis})

            print("finish half")
            savemat(save_path+'/storepayload_x.mat', mdict={'arr': env.payload_x})
            savemat(save_path+'/storepayload_y.mat', mdict={'arr': env.payload_y})
            savemat(save_path+'/storepayload_z.mat', mdict={'arr': env.payload_z})
            # savemat(save_path+'/storepos_z.mat', mdict={'arr': env.pos_z})
            # savemat(save_path+'/storevel_x.mat', mdict={'arr': env.vel_x})
            # savemat(save_path+'/storevel_y.mat', mdict={'arr': env.vel_y})
            # savemat(save_path+'/storepayload_vx.mat', mdict={'arr': env.payload_vx})
            savemat(save_path+'/storepayload_vy.mat', mdict={'arr': payload_vy})
            savemat(save_path+'/storepayload_vz.mat', mdict={'arr': payload_vz})
            # savemat(save_path+'/storeconcensus_v.mat', mdict={'arr': env.v_des})
            savemat(save_path+'/storepayload_orientation_x.mat', mdict={'arr': env.payload_orientation_x})
            savemat(save_path+'/storepayload_orientation_y.mat', mdict={'arr': env.payload_orientation_y})
            savemat(save_path+'/storepayload_orientation_z.mat', mdict={'arr': env.payload_orientation_z})
            savemat(save_path+'/storeatt.mat', mdict={'arr': env.att})

            print("finish data saving")

        env.shutdown()
        break
    env.step()  







