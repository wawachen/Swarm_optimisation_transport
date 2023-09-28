from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.envs.drone_bacterium_agent import Drone

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
class Drone_Env_bacterium:

    def __init__(self,env_name,num_agents):

        #Settings for video recorder 
        fps = 2
        # environment parameters
        self.time_step = 0

        # configure spaces
        self.num_a = num_agents

        self.safe_distance = 0.71
        self.x_limit_min = -15+self.safe_distance/2
        self.x_limit_max = 15-self.safe_distance/2
        self.y_limit_min = -15+self.safe_distance/2
        self.y_limit_max = 15-self.safe_distance/2
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(num_agents)]
        self.payload = Shape('Cylinder18')

        # for j, agent in enumerate(self.agents):
        #     agent.agent.panoramic_camera.set_resolution([1024,512])

        # self.random_position_spread()
        # self.random_position_spread()
        self.flock_spread()

        # self.cam = VisionSensor('Video_recorder')
        # fourcc = VideoWriter_fourcc(*'MP42')
        # self.video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))


    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone1.ttm'))
            model_handles.append(m1)

        return model_handles

    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance <= self.safe_distance:
            return 1
        else:
            return 0

    def flock_spread(self):
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(-5,5),random.uniform(-5,5),1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_drone_position()[0]
                vy = self.agents[i].agent.get_drone_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                vpt = [random.uniform(-5,5),random.uniform(-5,5)]
                check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)
                while check_conditions:
                    vpt = [random.uniform(-5,5),random.uniform(-5,5)]
                    check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                vpts.append(vpt)
                saved_agents.append(i)


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
    

    def step(self):
        self.time_step += 1
    
        for i,agent in enumerate(self.agents):
            if agent.agent.get_drone_position()[2]<1.0:
                print("rescue!")
                vels1 = agent.agent.position_controller([agent.agent.get_drone_position()[0],agent.agent.get_drone_position()[1],1.7])
                agent.agent.set_propller_velocity(vels1)

                agent.agent.control_propeller_thrust(1)
                agent.agent.control_propeller_thrust(2)
                agent.agent.control_propeller_thrust(3)
                agent.agent.control_propeller_thrust(4)
            
            else:
                vel = np.zeros(3)

                p_pos = agent.agent.get_drone_position()
                wall_dists = np.array([np.abs(15.0-p_pos[1]),np.abs(15.0+p_pos[1]),np.abs(15.0+p_pos[0]),np.abs(15.0-p_pos[0])])
                closest_wall = np.argmin(wall_dists)
                nearest_dis = wall_dists[closest_wall]

                agent.agent.depth_sensor.set_position([p_pos[0],p_pos[1],p_pos[2]-0.5])
                agent.agent.depth_sensor.set_orientation([-np.radians(180),0,0])

                agent.agent.panoramic_holder.set_position([p_pos[0],p_pos[1],p_pos[2]+0.1],reset_dynamics=False)
                agent.agent.panoramic_holder.set_orientation([0,0,np.radians(90)],reset_dynamics=False)

                agent.agent.proximity_holder.set_position([p_pos[0],p_pos[1],p_pos[2]+0.1],reset_dynamics=False)
                agent.agent.proximity_holder.set_orientation([0,0,0],reset_dynamics=False)
                # vel = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[:2]
                activate = agent.obstacle_detection()
                # activate = 0

                if activate == 1:
                    if agent.agent.get_detection(self.payload) == 1:
                        q_o = 1.2-agent.agent.get_concentration()
                        vel_obs = np.zeros(3)
                        obstacle_avoidance_velocity = agent.obstacle_avoidance1()
                        vel_obs[0] = 0.3*agent.flock_controller()[0] + obstacle_avoidance_velocity[0]/(q_o*150)+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                        vel_obs[1] = 0.3*agent.flock_controller()[1] + obstacle_avoidance_velocity[1]/(q_o*150)+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                        vel_obs[2] = obstacle_avoidance_velocity[2]

                        agent.set_action(vel_obs)
                        print("obs",i,vel_obs)
                    else:
                        vel_obs = np.zeros(3)
                        obstacle_avoidance_velocity = agent.obstacle_avoidance1()
                        vel_obs[0] = obstacle_avoidance_velocity[0]+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                        vel_obs[1] = obstacle_avoidance_velocity[1]+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                        vel_obs[2] = obstacle_avoidance_velocity[2]

                        agent.set_action(vel_obs)
                        # print("obs",i,vel_obs)

                else:
                    if agent.agent.get_detection(self.payload) == 1:
                        if agent.is_neighbour()>0:
                            # print("wawa")
                            vel[0] = 0.3*agent.flock_controller()[0] + agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                            vel[1] = 0.3*agent.flock_controller()[1] + agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                            vel[2] = agent.flock_controller()[2]
                            agent.set_action(vel)
                            
                        else:
                            # print("wawa1")
                            vel[0] = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                            vel[1] = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                            vel[2] = agent.flock_controller()[2]
                            agent.set_action(vel)
                    else:
                        # if agent.is_neighbour()>0:
                        #     # print("wawa2")
                        #     vel[0] = agent.flock_controller()[0] + agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                        #     vel[1] = agent.flock_controller()[1] + agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                        #     vel[2] = agent.flock_controller()[2]
                            
                        # else:       
                        vel[0] = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                        vel[1] = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1] 
                        vel[2] = agent.flock_controller()[2] 
                    
                        agent.set_action(vel)
                
                # vel = agent.flock_controller()
                    # print(i,vel)
                # print('bacterium',agent.bacterium_controller()[:2], 'flock', agent.flock_controller()[:2])
                # vel = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[:2]
                # print(agent.is_neighbour())
                # vel = agent.flock_controller()
                
                # agent.set_action(vel)
        
        self.pr.step()  #Step the physics simulation

        # img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.video.write(img_rgb)
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

   

