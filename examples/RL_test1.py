"""
An example of how one might use PyRep to create their RL environments.
In this case, the quadricopter must manipulate a target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
"""

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
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
from pyrep.objects.vision_sensor import VisionSensor

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
            minimum=[-2.5,-2.5, 0.0, -np.float32(np.pi),-np.float32(np.pi)/2,-np.float32(np.pi),-2.5,-2.5, 0.0, -np.float32(np.pi),-np.float32(np.pi)/2,-np.float32(np.pi),0.0],
            maximum=[2.5, 2.5, 3.0, np.float32(np.pi),np.float32(np.pi)/2,np.float32(np.pi), 2.5, 2.5, 3.0, -np.float32(np.pi),-np.float32(np.pi)/2,-np.float32(np.pi),3.0]
            )

        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()

        self.agent1 = Quadricopter()
        self.agent2 = Quadricopter()
        self.agent3 = Quadricopter()
        self.agent4 = Quadricopter()
        self.agent5 = Quadricopter()
        self.agent6 = Quadricopter()

        self.suction_cup = UarmVacuumGripper()
        self.target = Shape('Payload')
        self.cam = VisionSensor('Vision_sensor')
        self._episode_ended = False
        self.step_counter = 0
        

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_state(self):
        dis_suction_target = np.sqrt(np.power((self.target.get_position()[:]-self.suction_cup.get_suction_position()[:]),2).sum())
        vector_state = np.r_[self.target.get_position()[:], self.target.get_orientation()[:],self.agent.get_drone_position()[:], self.agent.get_orientation()[:],]
        vector_state1 = vector_state.astype(np.float32)
        return vector_state1

    def _reset(self):
        # Get a random position within a cuboid and set the target position 
        self.suction_cup.release()
        self.agent.drone_reset()

        self.target.set_position([0.0,0.0,0.25])
        self.target.set_orientation([0.0, 0.0, 0.0])
        self.agent.set_3d_pose([0.0,0.0,1.5,0.0,0.0,0.0])

        self._state = self._get_state()
        self._episode_ended = False
        self.step_counter = 0
        return ts.restart(self._state)

    def _step(self, action):
        
        if self._episode_ended:
            self.step_counter = 0
            return self.reset()

        self.step_counter+= 1

        # for act in action:
        #     if act<0.0 or act>11.1:
        #         raise ValueError('`action` should be between 0 and 11.1')

        self.agent.set_propller_velocity(action[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
        
        self.suction_cup.grasp(self.target)
        self.pr.step()  # Step the physics simulation

        self._state = self._get_state()
        detect = self.suction_cup.check_connection(self.target)
        pos_cur = self._state[6:9]
        target_cur = self._state[:3]
        orientation_cur = self._state[3:6]
        
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


#evaluation
def compute_avg_return(num, py_env, environment, policy, num_episodes=5,recording = False):   

  total_return = 0.0

  if recording:
      fps = 24
      fourcc = VideoWriter_fourcc(*'MP42')
      video = VideoWriter('./my_vid_%d.avi'%(num), fourcc, float(fps), (py_env.cam.get_resolution()[0], py_env.cam.get_resolution()[1]))

  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    
    if recording:
        img = (py_env.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img_rgb)
    
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)

      if recording:
        img = (py_env.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img_rgb)

      episode_return += time_step.reward
    total_return += episode_return
    

  avg_return = total_return / num_episodes
  if recording:
    video.release()
  return avg_return.numpy()[0]


#---------------------------------------------------------------------------------------------------------------
#parameters:
env_name = join(dirname(abspath(__file__)), 'cooperative_transportation_uav.ttt') # @param {type:"string"}
#env_name1 = join(dirname(abspath(__file__)), 'eval_cooperative_transportation_uav.ttt')
#eval_env_name = join(dirname(abspath(__file__)), 'eval_cooperative_transportation_uav.ttt')

# use "num_iterations = 1e6" for better results,
# 1e5 is just so this doesn't take too long. 
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"} 
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 1000000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}
gradient_clipping = None # @param

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 30 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}
#-----------------------------------------------------------------------------------------------------------------

#environment 
train_py_env = Drone_Env(env_name)
#eval_py_env = Drone_Env(eval_env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)

#eval_env = tf_py_environment.TFPyEnvironmehe report. The pack error shont(eval_py_env)
#print(train_env.time_step_spec())
#define agent
#critic net
observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()

# preprocessing_layers = {
#     'floor_image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 3),
#                                         tf.keras.layers.Conv2D(4, 3),
#                                         tf.keras.layers.Flatten()]),    #maybe we need dropout layer
#     'vector': tf.keras.layers.Dense(12)
#     }
# preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
# critic_net = critic_network.CriticNetwork(
#     (observation_spec, action_spec),
#     observation_fc_layer_params=None,
#     action_fc_layer_params=None,
#     joint_fc_layer_params=critic_joint_fc_layer_params)
# actor_net = ActorNetwork(observation_spec, 
#                       action_spec,
#                       preprocessing_layers=preprocessing_layers,
#                       preprocessing_combiner=preprocessing_combiner)
time_steptt = train_env.reset()
# b= actor_net(time_steptt.observation, time_steptt.step_type)
print(time_steptt)


critic_net = critic_network.CriticNetwork((observation_spec,action_spec), 
                   observation_fc_layer_params=None,
                   action_fc_layer_params=None,
                   joint_fc_layer_params=critic_joint_fc_layer_params
                   )


# time_steptt =  {
#     'floor_image': tf.random.uniform([1,32,32,3],dtype = np.float32),
#     'vector': tf.random.uniform([1, 12],dtype = np.float32)
#     }

#time_steptt = train_env.reset()
#print(time_steptt.observation)
#critic_net((time_steptt.observation,tf.random.uniform([1, 4],dtype = np.float32)))
# critic_net = ValueRnnNetwork(
#         observation_spec,
#         preprocessing_layers=preprocessing_layers,
#         preprocessing_combiner=preprocessing_combiner,
#         conv_layer_params=[(16, 8, 4), (32, 4, 2)],
#         input_fc_layer_params=(256,),
#         lstm_size=(256,),
#         output_fc_layer_params=(128,),
#         activation_fn=tf.keras.activations.relu)

#actor net
# actor_net = ActorNetwork(observation_spec, 
#                      action_spec,
#                      preprocessing_layers=preprocessing_layers,
#                      preprocessing_combiner=preprocessing_combiner)
# def normal_projection_net(action_spec,init_means_output_factor=0.1):
#   return normal_projection_network.NormalProjectionNetwork(
#       action_spec,
#       mean_transform=None,
#       state_dependent_std=True,
#       init_means_output_factor=init_means_output_factor,
#       std_transform=sac_agent.std_clip_transform,
#       scale_distribution=True)

# actor_net = actor_distribution_network.ActorDistributionNetwork(
#     observation_spec,
#     action_spec,
#     preprocessing_layers=preprocessing_layers,
#     preprocessing_combiner=preprocessing_combiner,
#     fc_layer_params=actor_fc_layer_params,
#     continuous_projection_net=normal_projection_net)

def normal_projection_net(action_spec,init_means_output_factor=0.1):
   return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layer_params,
    continuous_projection_net=normal_projection_net)


# actor_net = ActorDistributionRnnNetwork(
#         observation_spec,
#         action_spec,
#         preprocessing_layers=preprocessing_layers,
#         preprocessing_combiner=preprocessing_combiner,
#         input_fc_layer_params=(256,),
#         lstm_size=(256,),
#         output_fc_layer_params=(128,),
#         activation_fn=tf.keras.activations.relu)

#instantiate the agent
global_step = tf.compat.v1.train.get_or_create_global_step()
tf_agent = sac_agent.SacAgent(
    train_env.time_step_spec(),
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate),
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)

tf_agent.initialize()

#define policy
eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
collect_policy = tf_agent.collect_policy

#replay buffer 
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

#data collection 
initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)

initial_collect_driver.run()

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)

#train the agent 
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)
collect_driver.run = common.function(collect_driver.run)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
ac1 = actor_policy.ActorPolicy(
        time_step_spec=train_env.time_step_spec(),
        action_spec=action_spec,
        actor_network=tf_agent._actor_network,
        training=False)

policies = [ac1]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_driver.run()

  # Sample a batch of data from the buffer again for output formatting and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = tf_agent.train(experience)

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    #compute_avg_return(return_value, eval_env, eval_policy, num_eval_episodes)
    ac1 = actor_policy.ActorPolicy(
        time_step_spec=train_env.time_step_spec(),
        action_spec=action_spec,
        actor_network=tf_agent._actor_network,
        training=False)
    policies.append(ac1)


train_py_env.shutdown()
#----------------------------------------------------------------------------------
#evaluation
eval_py_env = Drone_Env(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

steps = range(0, num_iterations + 1, eval_interval)
step_c = 0
returns = []
policy_len = len(policies)

for policy in policies:
    eval_policy = policy
    record_flag = False

    if (step_c == 0) or (step_c == round(policy_len/2)) or (step_c == (policy_len-1)):
        record_flag = True

    avg_return = compute_avg_return(step_c, eval_py_env, eval_env, eval_policy, num_eval_episodes,record_flag)
    print('step = {0}: Average Return = {1}'.format(steps[step_c], avg_return))
    step_c+= 1
    returns.append(avg_return)

#plot the reward

plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.savefig('./reward_diagram')

'''
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

width = 1280
height = 720
fps = 24
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./my_vid.avi', fourcc, float(fps), (width, height))

pr = PyRep()
pr.launch(...)
cam = VisionSensor('my_cam')

for _ in range(100):
    img = (cam.capture_rgb() * 255).astype(np.uint8)
    video.write(img)
video.release()
'''

print('Done!')

eval_py_env.shutdown()