from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
from pyrep.objects.vision_sensor import VisionSensor
import random
import math

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math

import torch
from pyrep.policies.utilities1 import soft_update, transpose_to_tensor, transpose_list, hard_update
from pyrep.policies.utilities1 import _relative_headings, _shortest_vec, _distances, _relative_headings, _product_difference, _distance_rewards, take_along_axis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,env_name,num_agents):

        #Settings for video recorder 
        fps = 2

        self.reset_callback = self.reset_world
        self.reward_callback = self.reward
        self.observation_callback = self.observation
        self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = True
        self.time_step = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents

        self.x = np.zeros((2, self.num_a))
        # Heading in range [0, 2π]
        self.theta = np.zeros(self.num_a)
        self.n_nearest = 5

        self.safe_distance = 0.71
        self.x_limit_min = -7.5+self.safe_distance/2
        self.x_limit_max = 7.5-self.safe_distance/2
        self.y_limit_min = -7.5+self.safe_distance/2
        self.y_limit_max = 7.5-self.safe_distance/2
        
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(num_agents)]
        self.target = Shape('Cuboid')
        self.xd = self.target.get_position()[:2]

        for i,agent in enumerate(self.agents):
            self.x[0,i] = agent.get_2d_pos()[0]
            self.x[1,i] = agent.get_2d_pos()[1]
            self.theta[i] = np.radians(agent.theta)

        d, obs_n = self.observation_callback(self.x, self.xd, self.theta, self.num_a, self.n_nearest)
        obs_dim = obs_n[0,:].shape[0]
        # print(obs_dim)

        for agent in self.agents:
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(5) #3*3
            # else:
            #     u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
           
            self.action_space.append(u_action_space) 
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        self.cam = VisionSensor('Video_recorder')
        fourcc = VideoWriter_fourcc(*'MP42')
        self.video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))


    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
            model_handles.append(m1)

        # [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_1,m1_1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_2,m1_2]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_3,m1_3]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_4,m1_4]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_5,m1_5]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))

        # model_handles = [m1,m1_1,m1_2,m1_3,m1_4,m1_5]
        return model_handles

    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance <= self.safe_distance:
            return 1
        else:
            return 0

    def spread(self):
        self.agents[0].agent.set_3d_pose([15,-6,1.7,0.0,0.0,0.0])
        self.agents[1].agent.set_3d_pose([15,0,1.7,0.0,0.0,0.0])
        self.agents[2].agent.set_3d_pose([15,6,1.7,0.0,0.0,0.0])
        self.agents[3].agent.set_3d_pose([20,-6,1.7,0.0,0.0,0.0])
        self.agents[4].agent.set_3d_pose([20,0,1.7,0.0,0.0,0.0])
        self.agents[5].agent.set_3d_pose([20,6,1.7,0.0,0.0,0.0])

    def random_position_spread(self):
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_drone_position()[0]
                vy = self.agents[i].agent.get_drone_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)
                while check_conditions:
                    vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                    check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                vpts.append(vpt)
                saved_agents.append(i)
    
    def drone_crash_recover(self,num,agent):
        self.pr.remove_model(self.model_handles[num])
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        self.model_handles[num] = m1
        self.agents[num] = Drone(num)

        vpts = []

        for i in range(self.num_a):
            if i == num:
                pass
            else:
                vx = self.agents[i].agent.get_drone_position()[0]
                vy = self.agents[i].agent.get_drone_position()[1]
                vpts.append([vx,vy])

        vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
        check_list = [self.check_collision(vpt,vv) for vv in vpts]
        check_conditions = np.sum(check_list)
        while check_conditions:
            vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
            check_list = [self.check_collision(vpt,vv) for vv in vpts]
            check_conditions = np.sum(check_list)

        self.agents[num].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
        self.agents[num]._reset()

        for i,agent in enumerate(self.agents):
            self.x[0,i] = agent.get_2d_pos()[0]
            self.x[1,i] = agent.get_2d_pos()[1]
            self.theta[i] = np.radians(agent.theta)

        d, obs_n = self.observation_callback(self.x, self.xd, self.theta, self.num_a, self.n_nearest)
        return obs_n[num,:]


    def action_spec(self):
        return self.action_space[0]

    def observation_spec(self):
        return self.observation_space[0]

    def translate_action1(self,agent,action):
        action_n = np.zeros(2)

        if action == 0:
            r = -90
            agent.theta_cmd = r
            agent.theta += r 
            agent.theta = np.mod(agent.theta,360)

            action_n[0] = agent.v*math.cos(np.radians(agent.theta))
            action_n[1] = agent.v*math.sin(np.radians(agent.theta))

        if action == 1:
            r = -45
            agent.theta_cmd = r
            agent.theta += r 
            agent.theta = np.mod(agent.theta,360)

            action_n[0] = agent.v*math.cos(np.radians(agent.theta))
            action_n[1] = agent.v*math.sin(np.radians(agent.theta))

        if action == 2:
            r = 0
            agent.theta_cmd = r
            agent.theta += r 
            agent.theta = np.mod(agent.theta,360)

            action_n[0] = agent.v*math.cos(np.radians(agent.theta))
            action_n[1] = agent.v*math.sin(np.radians(agent.theta))

        if action == 3:
            r = 45
            agent.theta_cmd = r
            agent.theta += r 
            agent.theta = np.mod(agent.theta,360)

            action_n[0] = agent.v*math.cos(np.radians(agent.theta))
            action_n[1] = agent.v*math.sin(np.radians(agent.theta))

        if action == 4:
            r = 90
            agent.theta_cmd = r
            agent.theta += r 
            agent.theta = np.mod(agent.theta,360)

            action_n[0] = agent.v*math.cos(np.radians(agent.theta))
            action_n[1] = agent.v*math.sin(np.radians(agent.theta))

        return action_n

    def reset_world(self):
        self.time_step = 0
        #self.suction_cup.release()
        for i in range(self.num_a):
            self.pr.remove_model(self.model_handles[i])

        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(self.num_a)]

        self.target.set_position([0.0,0.0,0.3])
        self.target.set_orientation([0.0, 0.0, 0.0])
        self.random_position_spread()

        for j in range(self.num_a):
            self.agents[j]._reset()

        img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.video.write(img_rgb)

        #for hovering when reset
        # pos = np.zeros([self.num_a,2])
        for j in range(10):
            for i in range(self.num_a):
                # if j == 0: 
                pos = self.agents[i].get_2d_pos()[:]
                self.agents[i].hover(pos)
            self.pr.step()

        for i,agent in enumerate(self.agents):
            self.x[0,i] = agent.get_2d_pos()[0]
            self.x[1,i] = agent.get_2d_pos()[1]
            self.theta[i] = np.radians(agent.theta)

        d, obs_n = self.observation_callback(self.x, self.xd, self.theta, self.num_a, self.n_nearest)

        return obs_n

    def stepPID(self, action_n, max_t):
        print(self.time_step)
        self.time_step += 1
        obs_n = []
        reward_n = []
        done_n = []

        for i,agent in enumerate(self.agents):
            action_n1 = self.translate_action1(agent,action_n[i])
            # print(action_n1)
            agent.set_action_pos(action_n1,self.pr)
            # self.pr.step()  #Step the physics simulation

        img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.video.write(img_rgb)
            
        # if self.step_counter>1200:
        #     self._episode_ended = True
        wall_coli = []
        des = []

        for i,agent in enumerate(self.agents):
            self.x[0,i] = agent.get_2d_pos()[0]
            self.x[1,i] = agent.get_2d_pos()[1]
            self.theta[i] = np.radians(agent.theta)
            wall_coli.append(agent.wall_detection())
            des.append(np.sqrt(sum((agent.get_2d_pos()[:]-self.target.get_position()[:2])**2)))
            done_n.append(self.done_callback(max_t))
        # record observation for each agent
        # print(self.n_nearest)
        d, obs_n = self.observation_callback(self.x, self.xd, self.theta, self.num_a, self.n_nearest)
        wall_coli = np.r_[wall_coli]
        # print(wall_coli.shape)
        des = np.array(des)
        # print(des.shape)
        reward_n = self.reward_callback(d, self.num_a, wall_coli, des)
    
        return obs_n, reward_n, np.array(done_n)

    def step1(self, action_n, max_t):
        self.time_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        action_n1 = np.zeros([self.num_a,2])

        for j in range(20):
            for i,agent in enumerate(self.agents):
                if j == 0:
                    action_n1[i,0] = self.translate_action1(agent,action_n[i])[0]
                    action_n1[i,1] = self.translate_action1(agent,action_n[i])[1]
                agent.set_action(action_n1[i,:])
            self.pr.step()  #Step the physics simulation

        img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.video.write(img_rgb)
        
        # pos1 = np.zeros([self.num_a,2])
        for j in range(40):
            for i,agent in enumerate(self.agents):
                # if j == 0:
                # pos1[i,0] = agent.get_2d_pos()[0]
                # pos1[i,1] = agent.get_2d_pos()[1]
                pos1 = agent.get_2d_pos()[:]
                agent.hover(pos1)
            self.pr.step()
            
        # if self.step_counter>1200:
        #     self._episode_ended = True
        wall_coli = []
        des = []

        for i,agent in enumerate(self.agents):
            self.x[0,i] = agent.get_2d_pos()[0]
            self.x[1,i] = agent.get_2d_pos()[1]
            self.theta[i] = np.radians(agent.theta)
            wall_coli.append(agent.wall_detection())
            des.append(np.sqrt(sum((agent.get_2d_pos()[:]-self.target.get_position()[:2])**2)))
            done_n.append(self.done_callback(max_t))
        # record observation for each agent
        # print(self.n_nearest)
        d, obs_n = self.observation_callback(self.x, self.xd, self.theta, self.num_a, self.n_nearest)
        wall_coli = np.r_[wall_coli]
        # print(wall_coli.shape)
        des = np.array(des)
        # print(des.shape)
        reward_n = self.reward_callback(d, self.num_a, wall_coli, des)
    
        return obs_n, reward_n, np.array(done_n)
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def reward(self, d, n_agents, wall_coli, des):
        """
        Get rewards for each agent based on distances to other boids

        Args:
            d (np.array): 2d array representing euclidean distances between
                each pair of boids

        Returns:
            np.array: 1d array of reward values for each agent
        """
        proximity_threshold = 0.8
        distant_threshold = 1.5

        dis_rewards = _distance_rewards(
            d, proximity_threshold, distant_threshold,
        )

        wall_coli1 = np.any(wall_coli,axis=1)
        func = lambda x: -1000 if x==1 else 0 
        wall_penalties = [func(i) for i in wall_coli1]
        wall_penalties = np.array(wall_penalties)

        # func1 = lambda x: 1000 if x[0]<6.2 and x[0]>-3.8 and x[1]<2 and x[1]>-2 else 0
        # goal_rewards =  [func1(agent.get_2d_pos()[:]) for agent in self.agents]
        # goal_rewards = np.array(goal_rewards)

        func1 = lambda x: 1000 if x<5 else 0
        goal_rewards =  [func1(i) for i in des]
        goal_rewards = np.array(goal_rewards)


        crash_penalties = [-agent.crash_detection()*10000 for agent in self.agents]
        crash_penalties = np.array(crash_penalties)

        return dis_rewards + wall_penalties + goal_rewards + crash_penalties
    
    def done(self, t_max):
        # if target_d < 0.1 or np.any(agent.wall_collision) or world.timestep == t_max:
        # if np.any(agent.wall_collision) or world.timestep == t_max:
        if self.time_step == t_max:
           return 1
        else:
           return 0

    def get_distance(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def get_bearing(self,p1,p2):
        ang = math.atan2(p2[0]-p1[0], p2[1]-p1[1])

        if ang<=-np.pi/2:
           return ang + 3*np.pi/2
        else:
           return ang - np.pi/2
        
    def get_turning_angle(self,a,b):
        a1 = np.sqrt((a[0] * a[0]) + (a[1] * a[1]))
        b1 = np.sqrt((b[0] * b[0]) + (b[1] * b[1]))
        aXb = (a[0] * b[0]) + (a[1] * b[1])

        cos_ab = aXb/(a1*b1)
        angle_ab = math.acos(cos_ab)*(180.0/np.pi)
        return angle_ab

    def point_direction(self,a,b,c):
        #start, end , point S = (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3) 
        #S>0, left; S<0, right; S=0, on the line
        S = (a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0])
        if S < 0:
            return 1
        if S > 0:
            return -1
        if S == 0:
            return 1

    def agent_turning_angle(self, agent, world):
        a = world.landmarks[0].state.p_pos - agent.state.p_pos
        b = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
        angle = self.get_turning_angle(a,b)
        a1 = agent.state.p_pos
        b1 = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
        c1 = world.landmarks[0].state.p_pos

        return angle*self.point_direction(a1,b1,c1)


    def observation(self, x, xd, theta, n_agents, n_nearest) -> np.array:
        """
        theta is radians
        Returns a view on the flock phase space local to each agent. Since
        in this case all the agents move at the same speed we return the
        x and y components of vectors relative to each boid and the relative
        heading relative to each agent.

        In order for the agents to have similar observed states, for each agent
        neighbouring boids are sorted in distance order and then the closest
        neighbours included in the observation space

        Returns:
            np.array: Array of local observations for each agent, bounded to
                the range [-1,1]
        """
        xs = _product_difference(x[0], n_agents)
        ys = _product_difference(x[1], n_agents)
        d = _distances(xs, ys)
        
        # Sorted indices of flock members by distance
        sort_idx = np.argsort(d, axis=1)[:, : n_nearest]

        #print(self.theta.shape)
        relative_headings = _relative_headings(theta)
        # print(relative_headings)

        closest_x = take_along_axis(xs, sort_idx) #np.sort(a,axis=1)
        closest_y = take_along_axis(ys, sort_idx)
        closest_h = take_along_axis(relative_headings, sort_idx)
        # print(closest_y)
        # Rotate relative co-ords relative to each boids heading
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        
        x1 = (cos_t * closest_x + sin_t * closest_y) / 21
        y1 = (cos_t * closest_y - sin_t * closest_x) / 21

        des_xs = (xd[0] - x[0])[:,np.newaxis]
        des_ys = (xd[1] - x[1])[:,np.newaxis]

        des_x = (cos_t * des_xs + sin_t * des_ys) / 10.5
        des_y = (cos_t * des_ys - sin_t * des_xs) / 10.5

        #walls obstacle
        wall_dists = []
        wall_angles = []
        for i in range(n_agents):
            wall_dists.append(np.array([np.abs(7.5-self.agents[i].get_2d_pos()[1]),np.abs(7.5+self.agents[i].get_2d_pos()[1]),np.abs(7.5+self.agents[i].get_2d_pos()[0]),np.abs(7.5-self.agents[i].get_2d_pos()[0])])/15) # rangefinder: forward, back, left, right
            wall_angles.append(np.array([np.pi / 2, 3 / 2 * np.pi, np.pi, 0.0]) - self.theta[i])
        #print(len(entity_dis),len(wall_dists))
        wall_dists = np.r_[wall_dists]
        wall_angles = np.r_[wall_angles]
        # print(wall_angles)
        
        wall_obs = np.zeros([n_agents,3])
        closest_wall = np.argmin(wall_dists,axis=1)

        for j in range(n_agents):
            wall_obs[j,0] = wall_dists[j,closest_wall[j]]
            wall_obs[j,1] = np.cos(wall_angles[j,closest_wall[j]])
            wall_obs[j,2] = np.sin(wall_angles[j,closest_wall[j]])

    
        local_observation = np.concatenate(
            [x1, y1, closest_h, des_x, des_y, wall_obs], axis=1
        )
        
        return d, local_observation



