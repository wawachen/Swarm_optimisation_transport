from os import path
from os.path import dirname, join, abspath

from numpy.core.defchararray import add
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
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        self.env_name = env_name
        # self.close_simulation = False

        self.safe_distance = 0.71
        self.x_limit_min = -7.5+self.safe_distance/2
        self.x_limit_max = 7.5-self.safe_distance/2
        self.y_limit_min = -7.5+self.safe_distance/2
        self.y_limit_max = 7.5-self.safe_distance/2

        self.shared_reward = True
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(num_agents)]
        # self.reset_world()
        self.payload = Shape('Cylinder18')
        self.goals = self.generate_goal()
        self.enterlift = 0
        self.enterhover = 0
        self.pos_des = np.zeros([num_agents,3])
        self.pos_des1 = np.zeros([num_agents,3])
        self.concen_collect = np.zeros(num_agents)

        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(5) #3*3
            else:
                u_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            #observation space
            obs_dim = len(self.observation_callback(agent))
            # print(obs_dim)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
            model_handles.append(m1)

        # [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_1,m1_1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_2,m1_2]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_3,m1_3]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_4,m1_4]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_5,m1_5]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))

        # model_handles = [m1,m1_1,m1_2,m1_3,m1_4,m1_5]
        return model_handles

    
    def check_collision_a(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return True if dist <= self.safe_distance else False

    def check_collision_p(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 1.0:
            return 1
        else:
            return 0

    def generate_goal(self):
        #visualization goal
        targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]
        # targets = [Shape('goal'), Shape('goal0'), Shape('goal1')]
        goal_points = np.array([[-3,0,1.7],[3,0,1.7],[0,3,1.7]]) #[-3,0,1.7],[3,0,1.7],[0,3,1.7]

        for i in range(self.num_a):
            targets[i].set_position(goal_points[i])

        return goal_points

    def random_spread_without_obstacle(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone(i))
            
            vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
            self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
            check_conditions = self.agents[i].agent.assess_collision()

            while check_conditions:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                check_conditions = self.agents[i].agent.assess_collision()

            # self.agents[i].get_agent((vpt[0],vpt[1]))
            # print("all",vpts)
            # print("current",vpt)
            
        return model_handles,objs

    
    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        for i in range(self.num_a):
            self.pr.remove_model(self.model_handles[i])
        # self.model_handles = self.import_agent_models()
        # # self.agents = [Drone(i) for i in range(self.num_a)]
        # self.random_position_spread()

        self.model_handles,ii = self.random_spread_without_obstacle()
        # self.goals = self.generate_goal()

        for j in range(self.num_a):
            self.agents[j]._reset()
    
        #for hovering when reset
        for j in range(50):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
     
        obs_n = []
        for agent in self.agents:
            obs_n.append(self.observation_callback(agent))


        return obs_n


    def step(self, action_n):
        finish_sig = np.zeros(self.num_a)
        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.35))
        
        if np.all(finish_sig) or self.enterhover:
            #---------------transport------------------------------
            for i in range(self.num_a):
                detect = self.agents[i].agent.suction_cup.grasp(self.payload)
        
            if self.time_step > 200:
                if self.enterlift == 0:
                    for i in range(self.num_a):
                        p_pos = self.agents[i].agent.get_drone_position()
                        self.pos_des1[i,0] = p_pos[0]
                        self.pos_des1[i,1] = p_pos[1]
                        self.pos_des1[i,2] = 2.0

                ee = np.zeros(self.num_a)
                for i in range(self.num_a):
                    # flock_vel = self.agents[i].flock_controller()
                    vels = self.agents[i].agent.position_controller1(self.pos_des1[i,:])

                    self.agents[i].agent.set_propller_velocity(vels)

                    self.agents[i].agent.control_propeller_thrust(1)
                    self.agents[i].agent.control_propeller_thrust(2)
                    self.agents[i].agent.control_propeller_thrust(3)
                    self.agents[i].agent.control_propeller_thrust(4)
                
                self.enterlift = 1

            else:
                if self.time_step<100:
                    for agent in self.agents:
                        agent.hover(1.7)
                else:
                    if self.enterhover == 0:
                        for i in range(self.num_a):
                            p_pos = self.agents[i].agent.get_drone_position()
                            obj_h = self.agents[i].agent.get_concentration(self.payload)
                            self.concen_collect[i] = obj_h
                            des_h = p_pos[2] - (self.agents[i].agent.suction_cup.get_suction_position()[2]-obj_h)+0.0565+0.02+0.0019-0.01-0.1
                            self.pos_des[i,0] = p_pos[0]
                            self.pos_des[i,1] = p_pos[1]
                            self.pos_des[i,2] = des_h
                            print("agent",i,": ",obj_h)
                    self.enterhover = 1
                    for i in range(self.num_a):
                        # flock_vel = self.agents[i].flock_controller()
                        vels = self.agents[i].agent.position_controller1(self.pos_des[i,:])
                        self.agents[i].agent.set_propller_velocity(vels)
                        self.agents[i].agent.control_propeller_thrust(1)
                        self.agents[i].agent.control_propeller_thrust(2)
                        self.agents[i].agent.control_propeller_thrust(3)
                        self.agents[i].agent.control_propeller_thrust(4)

            # print("time",self.time_step)
            self.pr.step()
            self.time_step+=1
            obs_n = []
            for i, agent in enumerate(self.agents):
                #----------------------------
                obs_n.append(np.zeros(self.num_a*2+(self.num_a-1)*4))

            return np.array(obs_n),0, 0
            #------------------------------------------------------
        else:
            obs_n = []
            reward_n = []
            done_n = []

            for j in range(10):
                # set action for each agent
                for i, agent in enumerate(self.agents):
                    agent.set_action(action_n[i])
                    # agent.set_action_pos(action_n[i], self.pr)
                self.pr.step()

            # record observation for each agent
            for i, agent in enumerate(self.agents):
                #----------------------------
                obs_n.append(self.observation_callback(agent))
                rw,ter = self.reward_callback(agent)
                reward_n.append(rw)
                done_n.append(ter)

            #all agents get total reward in cooperative case
            reward = np.sum(reward_n[:]) #need modify
            if self.shared_reward:
                reward_n = [reward] * self.num_a

            # self.time_step+=1

            return np.array(obs_n), np.array(reward_n), np.array(done_n)


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def restart(self):
        if self.pr.running:
            self.pr.stop()
        self.pr.shutdown()

        self.pr = PyRep()
        self.pr.launch(self.env_name, headless=False)
        self.pr.start()
        # self.close_simulation = True
        

    def reward_and_terminate(self, agent):
        rew = 0
        terminate = 0
        finish_sig = np.zeros(self.num_a)

        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.1))
            rew -= min(dists)
        
        if np.all(finish_sig):
            terminate = 1
            rew = 0
            rew += 1

        #collision detection
        # wall_dists = np.array([np.abs(7.5-agent.agent.get_position()[1]),np.abs(7.5+agent.agent.get_position()[1]),np.abs(7.5+agent.agent.get_position()[0]),np.abs(7.5-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
        # wall_sig = np.any(wall_dists<0.206)

        # agent_collision = []
        # for a in self.agents:
        #     if a == agent: continue
        #     if self.check_collision_a(agent,a):
        #         agent_collision.append(1)
        #     else:
        #         agent_collision.append(0)
        # agent_sig = np.any(np.array(agent_collision))

        # if agent_sig or wall_sig:
        #     rew-=1
        #     terminate = 1
        if agent.agent.assess_collision():
            rew-=1
            terminate = 1
        
        if agent.agent.get_position()[2]<1.3:
            terminate = 1

        return rew,terminate


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

    # def agent_turning_angle(self, agent, world):
    #     a = world.landmarks[0].state.p_pos - agent.state.p_pos
    #     b = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
    #     angle = self.get_turning_angle(a,b)
    #     a1 = agent.state.p_pos
    #     b1 = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
    #     c1 = world.landmarks[0].state.p_pos

    #     return angle*self.point_direction(a1,b1,c1)


    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(self.goals.shape[0]):  # world.entities:
            entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/15)   
        # entity_pos.append(agent.target-agent.state.p_pos)
        
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/15.0)

        other_vel = []
        for other in self.agents:
            if other is agent: continue
            other_vel.append(other.get_2d_vel()-agent.get_2d_vel())

        # pos_obs = agent.agent.get_position()[:2]/2.5

        return np.concatenate(entity_pos + other_vel + other_pos)



