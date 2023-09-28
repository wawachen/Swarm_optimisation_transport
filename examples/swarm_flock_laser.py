from os import path
from os.path import dirname, join, abspath
from scipy.io import savemat
import signal
from mpi4py import MPI
from pyrep.common.arguments import get_args
import os

####################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# get the params
args = get_args()

if args.scenario_name=="circle":
    from pyrep.envs.drone_bacterium_env_laser import Drone_Env_bacterium
    SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testflock_circle2.ttt')
if args.scenario_name=="square":
    from pyrep.envs.drone_bacterium_env_laser import Drone_Env_bacterium
    SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testflock_square2.ttt')
if args.scenario_name=="peanut":
    from pyrep.envs.drone_bacterium_env_laser import Drone_Env_bacterium
    SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testflock_peanut2.ttt')
####################################

num_agents = 20
# create multiagent environment
env = Drone_Env_bacterium(args,rank,SCENE_FILE,num_agents)

save_path = './'+ 'flock_analysis/laser/'+args.scenario_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

    # signal.pause()
while 1:
    if env.done:
        pos_x = comm.gather(env.pos_x, root=0)
        pos_y = comm.gather(env.pos_y, root=0)
        pos_z = comm.gather(env.pos_z, root=0)
        c_sum = comm.gather(env.c_sum, root=0)

        if rank ==0:
            savemat(save_path+'/storePX.mat', mdict={'arr': pos_x})
            savemat(save_path+'/storePY.mat', mdict={'arr': pos_y})
            savemat(save_path+'/storePZ.mat', mdict={'arr': pos_z})

            savemat(save_path+'/storeAttachPoints.mat', mdict={'arr': env.attach_points})
            savemat(save_path+'/storeConcentration_sum.mat', mdict={'arr': c_sum})

            print("fail number", env.fail_num)
            print("finish data saving")

        env.shutdown()
        break
    env.step()  





