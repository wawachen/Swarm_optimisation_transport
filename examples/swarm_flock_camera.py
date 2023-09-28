from os.path import dirname, join, abspath
import numpy as np
from pyrep.common.arguments import get_args
import os
from tensorboardX import SummaryWriter
import math
from scipy.io import savemat

# get the params
args = get_args()

# if args.scenario_name=="circle":
# from pyrep.envs.drone_bacterium_env_camera import Drone_Env_bacterium
#     SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testflock_circle2.ttt')
# if args.scenario_name=="square":
from pyrep.envs.drone_bacterium_env_square import Drone_Env_bacterium
#     SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testflock_square2.ttt')
# if args.scenario_name=="peanut":
#     from pyrep.envs.drone_bacterium_env_peanut import Drone_Env_bacterium
# if args.field_size == 10:
#     env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
# if args.field_size == 15:
#     env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')
SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testflock_square2.ttt')

num_agents = 6
args.n_agents = num_agents
args.load_type = "six"
args.evaluate = True
# create multiagent environment
env = Drone_Env_bacterium(args, SCENE_FILE,num_agents)

save_path = './'+ 'flock_analysis/camera_{}agents'.format(num_agents)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not args.evaluate:
    # signal.pause()
    log_path = save_path+"/log_drone_MPI_V0_scratch_evaluation_table"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    returns = []
    time_f = []
    s_r = []
    deviation = []
    a_vel = []

    t_c = 0
    t_c_ex = 0

    for episode in range(20):
        # reset the environment
        if episode>0:
            env.reset_world()
        rewards = 0
        succ_list = []
        pos_x = []
        pos_y = []
        
        vel_mag1 = []
        
        while 1:
            vel_mag = []
            actions = []
            
            r, succ,pos,v = env.step()
            succ_list.append(succ)
            for j in range(args.n_agents):
                logger.add_scalar('Agent%d/pos_x'%j, pos[0][j,0], t_c)
                logger.add_scalar('Agent%d/pos_y'%j, pos[1][j,0], t_c)
                logger.add_scalar('Agent%d/vel_x'%j, v[0][j,0], t_c)
                logger.add_scalar('Agent%d/vel_y'%j, v[1][j,0], t_c)
                pos_x.append(pos[0][j,0])
                pos_y.append(pos[1][j,0])    
                vel_mag.append(math.sqrt(v[0][j,0]**2+v[1][j,0]**2))

            vel_mag1.append(np.sum(np.array(vel_mag))/len(vel_mag))  
            
            rewards += r[0]
        
            t_c += 1

            if np.sum(succ_list)==10 or env.done:
                break
            
        logger.add_scalar('Success rate', np.any(succ_list), episode)
        logger.add_scalar('Finish time', t_c-1 ,episode)
        logger.add_scalar('Rewards', rewards, episode)
        returns.append(rewards)
        
        if np.any(succ_list):
            time_f.append(t_c-t_c_ex)
            a_vel.append(np.sum(np.array(vel_mag1))/len(vel_mag1))
            # tt += (t_c-t_c_ex)
        t_c_ex = t_c
        s_r.append(np.any(succ_list))

        if np.any(succ_list):
            pos_x_end = pos_x[-args.n_agents*10:]
            pos_y_end = pos_y[-args.n_agents*10:]
            if args.n_agents == 3:
                goals = np.array([[2.8,0],[-2.8,0],[0,0]])
            if args.n_agents == 4:
                goals = np.array([[1.25,1.25],[-1.25,1.25],[1.25,-1.25],[-1.25,-1.25]])
            if args.n_agents == 6:
                goals = np.array([[2.25,-2.25],[-2.25,-2.25],[1.2,0],[-1.2,0],[2.25,2.25],[-2.25,2.25]])
            
            dev_all = []
            for m in range(10):
                d_e_t =  0
                for i in range(args.n_agents):
                    d_e = np.zeros(args.n_agents)
                    for j in range(args.n_agents):
                        d_e[j] = np.sqrt(np.sum((goals[i,:]-np.array([pos_x_end[args.n_agents*m:args.n_agents*m+args.n_agents][j],pos_y_end[args.n_agents*m:args.n_agents*m+args.n_agents][j]]))**2))
                    d_e_t += np.min(d_e)
                d_e_ta = d_e_t/args.n_agents
                dev_all.append(d_e_ta)
            deviation.append(np.sum(np.array(dev_all))/10)

        print('Returns is', rewards)
    print("Results is")
    print("Finished time: ", np.sum(np.array(time_f))/len(time_f), ', ', np.std(np.array(time_f)) )
    print('Average speed: ', np.sum(np.array(a_vel))/len(a_vel), ', ', np.std(np.array(a_vel)))
    print("Success rate: ", np.sum(np.array(s_r))/len(s_r))
    print("Deviation: ", np.sum(np.array(deviation))/len(deviation),', ', np.std(np.array(deviation)))
    print("Total rewards", np.sum(np.array(returns)))
    logger.close()
    env.shutdown()
else:
    succ_list = []
    
    posx_a = []
    posy_a = []
    
    while 1:
        vel_mag = []
        actions = []
        pos_x = np.zeros([args.n_agents,1])
        pos_y = np.zeros([args.n_agents,1])
        
        r, succ,pos,v = env.step()
        succ_list.append(succ)
        for j in range(args.n_agents):
            pos_x[j,0] = pos[0][j,0]
            pos_y[j,0] = pos[1][j,0]   
        posx_a.append(pos_x)
        posy_a.append(pos_y) 

        if np.sum(succ_list)==10 or env.done:
            break
    
    mdict = {'x':posx_a,'y':posy_a}
    savemat("data_pos{0}.mat".format(args.n_agents),mdict)
        
    env.shutdown()










