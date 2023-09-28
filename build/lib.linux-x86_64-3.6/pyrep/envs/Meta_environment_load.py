import numpy as np
import gym

from gym import spaces
from gym.utils import seeding
import math
import random
import yaml


class Load_depth_Env(gym.Env):
    """
        a depth image for the load 
    """
    def __init__(self, agent_num, task={}):
        super(Load_depth_Env, self).__init__()

        self.agent_num = agent_num
        load_depth_index = []
        self.load_depth = []
        load_name = ['depth_circle.npy','depth_square.npy','depth_peanut.npy']

        #pixel is 128x128
        for name in load_name:
            with open(name, 'rb') as f:
                depth_fl = np.load(f)
            ground_depth = depth_fl.min()
            load_depth_index.append(np.where(depth_fl>ground_depth))
            depth_fl[np.where(depth_fl<ground_depth)] = 0
            self.load_depth.append(depth_fl)
        
        self.kl_targets = [] 
        ld_num = 20
        for ld in load_depth_index:
            ld_r = math.floor(ld[0].shape[0]/ld_num)
            target = [] # ld_num+1 points
            for i in range(ld_num+1):
                target.append([ld[0][ld_r*i],ld[1][ld_r*i]])
            self.kl_targets.append(target)

        self.kl_targets = np.array(self.kl_targets) #[n_depth, 21, 2]

        #observation is the pixel location[2] and its depth value[1],and direction[2]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(5*agent_num,), dtype=np.float32)
        #action is move forward or turn right
        self.action_space = spaces.Discrete(2**agent_num)

        self._task = task
        self._goal = task.get('goals', np.zeros([ld_num+1,2]))
        self._map = task.get('maps', np.zeros([128,128]))
        self._state = np.zeros(5*agent_num, dtype=np.float32)
        self.seed()
        self.time_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        choice_num = random.sample(range(self.kl_targets.shape[0]),k=num_tasks)
        goals = self.kl_targets[choice_num]
        maps = self.load_depth[choice_num]
        tasks = [{'goals': goal, 'maps': map_s} for goal,map_s in zip(goals,maps)] #goals num_tasksx21x2
        
        return tasks

    def reset_task(self, task, map):
        self._task = task
        self._goal = task['goals']
        self._map = task['maps']

    def reset(self, env=True):
        self.time_step = 0
        self._state = np.zeros(5*self.agent_num, dtype=np.float32)

        for i in range(self.agent_num):
            self._state[i*5] = 20*i 
            self._state[i*5+1] = 0
            self._state[i*5+2] = self._map[self._state[i*5]][self._state[i*5+1]]
            self._state[i*5+3] = 1
            self._state[i*5+4] = 1
            
        return self._state

    def step(self, action):
        assert self.action_space.contains(action)
        self.time_step += 1
        #[ move forward (0) or turn right (1) ] [0 1 0 1 0 1]

        for i in range(self.agent_num):
            if action[i] == 0:
                if self._state[(i+1)*5-2] == 0 and self._state[(i+1)*5-1] == 0: #up 00
                    if self._state[i*5]-1>=0:
                        self._state[i*5] = self._state[i*5]-1
                        self._state[i*5+2] = self._map[self._state[i*5]][self._state[i*5+1]]
                if self._state[(i+1)*5-2] == 0 and self._state[(i+1)*5-1] == 1: #down 01
                    if self._state[i*5]+1<=127:
                        self._state[i*5] = self._state[i*5]+1
                        self._state[i*5+2] = self._map[self._state[i*5]][self._state[i*5+1]]
                if self._state[(i+1)*5-2] == 1 and self._state[(i+1)*5-1] == 0: #left 10
                    if self._state[i*5+1]-1>=0:
                        self._state[i*5+1] = self._state[i*5+1]-1
                        self._state[i*5+2] = self._map[self._state[i*5]][self._state[i*5+1]]
                if self._state[(i+1)*5-2] == 1 and self._state[(i+1)*5-1] == 1: #right 11
                    if self._state[i*5+1]+1<=127:
                        self._state[i*5+1] = self._state[i*5+1]+1
                        self._state[i*5+2] = self._map[self._state[i*5]][self._state[i*5+1]]
            elif action[i] == 1:
                if self._state[(i+1)*5-2] == 0 and self._state[(i+1)*5-1] == 0: #up
                   self._state[(i+1)*5-2] = 1
                   self._state[(i+1)*5-1] = 1
                if self._state[(i+1)*5-2] == 0 and self._state[(i+1)*5-1] == 1: #down
                   self._state[(i+1)*5-2] = 1
                   self._state[(i+1)*5-1] = 0
                if self._state[(i+1)*5-2] == 1 and self._state[(i+1)*5-1] == 0: #left
                   self._state[(i+1)*5-2] = 0
                   self._state[(i+1)*5-1] = 0
                if self._state[(i+1)*5-2] == 1 and self._state[(i+1)*5-1] == 1: #right
                   self._state[(i+1)*5-2] = 0
                   self._state[(i+1)*5-1] = 1

        kl_s = 0

        for i in range(self._goal.shape[0]): #iterate all points
            kl_vv = 0
            for j in range(-6,7):
                for k in range(-6,7):
                    if self._goal[i,:][0]+j>127 or self._goal[i,:][0]+j<0 or self._goal[i,:][1]+k>127 or self._goal[i,:][1]+k<0:
                        continue
                    kl_vv += self._map[self._goal[i,:][0]+j, self._goal[i,:][1]+k]
            kl_p = kl_vv/np.sum(self._map)
            
            kl_vv1 = 0
            x_lim_min = self._goal[i,:][0]-6
            x_lim_max = self._goal[i,:][0]+6
            y_lim_min = self._goal[i,:][1]-6
            y_lim_max = self._goal[i,:][1]+6

            if x_lim_min < 0:
                while x_lim_min < 0:
                    x_lim_min += 1
            if x_lim_max > 127:
                while x_lim_max > 127:
                    x_lim_max -= 1
            if y_lim_min < 0:
                while y_lim_min < 0:
                    y_lim_min += 1
            if y_lim_max > 127:
                while y_lim_max > 127:
                    y_lim_max -= 1
            
            for m in range(self.agent_num):
                p_x = self._state[m*5]
                p_y = self._state[m*5+1]

                if (p_x >= x_lim_min) and (p_x <= x_lim_max) and (p_y >= y_lim_min) and (p_y <= y_lim_max) :
                    kl_vv1 += 1

            kl_pp = kl_vv1/self.agent_num
            
            kl_s += -kl_p*np.log(kl_pp/kl_p)  

        reward = -kl_s

        if self.time_step >799:
            done = 1
        else:
            done = 0

        return self._state, reward, done, {'task': self._task}

if __name__=='__main__':
    from gym.envs.registration import register
    
    register(
    'Load_points-v0',
    entry_point='pyrep.envs.Meta_environment_load:Load_depth_Env',
    max_episode_steps=800
    )

    with open('/home/wawa/RL_transport_3D/pyrep/common/meta_mass_adaption.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()
