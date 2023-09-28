from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.quadricopter import Quadricopter
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
from pyrep.robots.end_effectors.uarm_Vacuum_Gripper import UarmVacuumGripper
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.ddpg import critic_network
#from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import normal_projection_network

from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils

from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from multiprocessing import Process,Value

tf.compat.v1.enable_v2_behavior()


class Drone_Env(py_environment.PyEnvironment):

    def __init__(self,env_name):
        #action: velocities of four rotors      
        self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=0.0, maximum=11.1)
        #observations: floor image, vector(pose of uav and payload), vector1(connecting state)
        self._observation_spec = array_spec.BoundedArraySpec(
            (13,),
            np.float32,
            minimum=[-2.5,-2.5,-1.5,-np.float32(np.pi),-np.float32(np.pi)/2,-np.float32(np.pi),-2.5,-2.5, 0.0, 0.0, 0.0, 0.0,0.0],
            maximum=[2.5,2.5,1.5,np.float32(np.pi),np.float32(np.pi)/2,np.float32(np.pi), 2.5, 2.5, 3.0, 11.1, 11.1, 11.1,11.1]
            )

        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()

        self.agent = Quadricopter()
        self.suction_cup = UarmVacuumGripper()
        self.target = Shape('Payload')
        self._episode_ended = False
        self.step_counter = 0


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_state(self):
        vector_state = np.r_[(self.target.get_position()[:]-np.array([0.0,0.0,1.5])), self.target.get_orientation()[:],self.agent.get_drone_position()[:], self.agent.velocities]
        return vector_state

    def _reset(self):
        # Get a random position within a cuboid and set the target position 
        self.suction_cup.release()

        self.target.set_position([0.0,0.0,0.25])
        self.agent.set_3d_pose([0.0,0.0,1.5,0.0,0.0,0.0])

        self._state = self._get_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        
        if self._episode_ended:
            self.step_counter = 0
            return self.reset()

        self.step_counter+= 1

        for act in action:
            if act<0.0 or act>11.1:
                raise ValueError('`action` should be between 0 and 1.')

        self.agent.set_propller_velocity(action[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
        
        self.suction_cup.grasp(self.target)
        self.pr.step()  # Step the physics simulation

        self._state = self._get_state()
        detect = self.suction_cup.check_connection(self.target)
        pos_cur = self._state[7:10]
        target_cur = self._state[:4]
        orientation_cur = self._state[4:7]
        
        drone_pos_limit = ((pos_cur[0]<-2.5) or (pos_cur[1]<-2.5) or (pos_cur[2]<0.0) or (pos_cur[0]>2.5) or (pos_cur[1]>2.5) or (pos_cur[2]>3.0))
        pyload_pos_limit = ((target_cur[0]<-2.5) or (target_cur[1]<-2.5) or (target_cur[2]<-1.5) or (target_cur[0]>2.5) or (target_cur[1]>2.5) or (target_cur[2]>1.5))
        reach_destination = (target_cur == np.array([0.0,0.0,0.0])).all()
        above_ground = (pos_cur[2]>0.5)

        if ((drone_pos_limit) or (pyload_pos_limit) or (reach_destination)or(self.step_counter>1200)):
            self._episode_ended = True
            if reach_destination:
                return ts.termination(self._state, reward = 1.0)
            elif (self.step_counter>1200):
                return ts.termination(self._state, reward = 0.0)
            else:
                return ts.termination(self._state, reward = -0.1)
        else:
            if above_ground:
                ground_reward = 0.25
            else:
                ground_reward = -0.25 

            #orientaion deviation
            orientation_reward =  0.2 if np.sqrt(np.power(orientation_cur,2).sum())<0.151 else -np.power(orientation_cur,2).sum()

            #suction cup deviation
            pos_dev = (target_cur+np.array([0.0,0.0,1.5])) - self.suction_cup.get_suction_position()
            suction_reward = -np.power(pos_dev,2).sum()
            
            reward = ground_reward-(np.power(target_cur[0],2)+np.power(target_cur[1],2)+np.power(target_cur[2],2))+orientation_reward+suction_reward+0.125*detect 
            return ts.transition(self._state, reward=reward, discount=1.0)

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


env_name = join(dirname(abspath(__file__)), 'cooperative_transportation_uav.ttt')
train_py_env = Drone_Env(env_name)

train_py_env.reset
#print(train_py_env._get_state())
target_cur = train_py_env._get_state()[:4]
print(target_cur)
reach_destination = (target_cur == np.array([0.0,0.0,0.0])).all()
print(reach_destination)

train_py_env.shutdown()