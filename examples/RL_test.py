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

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from pyrep.robots.mobiles.critic_net import CriticNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics

from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils

from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from multiprocessing import Process,Value

tf.compat.v1.enable_v2_behavior()


class ActorNetwork(network.Network):

    def __init__(self,
               observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='ActorNetwork'):
        super(ActorNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)

        self._action_spec = action_spec

        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        self._encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=False)
    
        initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        self._action_projection_layer = tf.keras.layers.Dense(
            action_spec.shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=initializer,
            name='action')

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        state, network_state = self._encoder(observations, step_type=step_type, network_state=network_state)
        print(state)
        # actions = self._action_projection_layer(state)
        # actions = common_utils.scale_to_spec(actions, self._action_spec)
        # actions = batch_squash.unflatten(actions)
        # return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state


class Drone_Env(py_environment.PyEnvironment):

    def __init__(self,env_name):
        #action: velocities of four rotors      
        self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=0.0, maximum=11.1)
        #observations: floor image, vector(pose of uav and payload), vector1(connecting state)
        self._observation_spec =  {
            'floor_image': array_spec.BoundedArraySpec((32, 32, 3), np.float32, minimum=0,
                                        maximum=255),
            'vector': array_spec.BoundedArraySpec((12,), np.float32, minimum=[-2.5,-2.5,0.0,-np.float32(np.pi),-np.float32(np.pi)/2,-np.float32(np.pi),-2.5,-2.5,0.0,-np.float32(np.pi),-np.float32(np.pi)/2,-np.float32(np.pi)],
                                        maximum=[2.5,2.5,3.0,np.float32(np.pi),np.float32(np.pi)/2,np.float32(np.pi),2.5,2.5,3.0,np.float32(np.pi),np.float32(np.pi)/2,np.float32(np.pi)])
           }

        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()

        self.agent = Quadricopter()
        self.suction_cup = UarmVacuumGripper()
        self.target = Shape('Payload')
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_state(self):
        vector_state = np.r_[self.agent.get_3d_pose(), self.target.get_position()[:], self.target.get_orientation()[:]]
        return [self.agent.get_picture()[0],vector_state]

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
            return self.reset()

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
        pos_cur = self._state[1][:3]
        target_cur = self._state[1][6:9]
        orientation_cur = self._state[1][9:]
        
        drone_pos_limit = ((pos_cur[0]<-2.5) or (pos_cur[1]<-2.5) or (pos_cur[2]<0.0) or (pos_cur[0]>2.5) or (pos_cur[1]>2.5) or (pos_cur[2]>3.0))
        pyload_pos_limit = ((target_cur[0]<-2.5) or (target_cur[1]<-2.5) or (target_cur[2]<0.0) or (target_cur[0]>2.5) or (target_cur[1]>2.5) or (target_cur[2]>3.0))
        reach_destination = (target_cur == np.array([0.0,0.0,1.5])).all()
        fall_down = (pos_cur[2] < 0.25)

        if ((drone_pos_limit) or (pyload_pos_limit) or (reach_destination)or(fall_down)):
            self._episode_ended = True
            if reach_destination:
                return ts.termination(self._state, reward = 0.0)
            else:
                return ts.termination(self._state, reward = -20.0)
        else:
            reward = -np.abs(target_cur[0])-np.abs(target_cur[1])-np.abs((target_cur[2]-1.5))-np.abs(orientation_cur[0])-np.abs(orientation_cur[1])-np.abs(orientation_cur[2])+10*detect 
            return ts.transition(self._state, reward=reward, discount=1.0)

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

#training
def run_step():
    train_py_env = Drone_Env(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

#evaluation
def compute_avg_return(return_value, policy, num_episodes=5):
    eval_py_env = Drone_Env(env_name1)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)  

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = eval_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return_value = avg_return.numpy()[0]
    eval_py_env.shutdown()

#---------------------------------------------------------------------------------------------------------------
#parameters:
env_name = join(dirname(abspath(__file__)), 'cooperative_transportation_uav.ttt') # @param {type:"string"}
env_name1 = join(dirname(abspath(__file__)), 'eval_cooperative_transportation_uav.ttt')
#eval_env_name = join(dirname(abspath(__file__)), 'eval_cooperative_transportation_uav.ttt')

# use "num_iterations = 1e6" for better results,
# 1e5 is just so this doesn't take too long. 
num_iterations = 1000 # @param {type:"integer"}

initial_collect_steps = 100 # @param {type:"integer"} 
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 1000 # @param {type:"integer"}

batch_size = 25 # @param {type:"integer"}

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

log_interval = 50 # @param {type:"integer"}

num_eval_episodes = 30 # @param {type:"integer"}
eval_interval = 100 # @param {type:"integer"}
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

preprocessing_layers = {
    'floor_image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 3),
                                        tf.keras.layers.Conv2D(4, 3),
                                        tf.keras.layers.Flatten()]),    #maybe we need dropout layer
    'vector': tf.keras.layers.Dense(12)
    }
preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
# critic_net = critic_network.CriticNetwork(
#     (observation_spec, action_spec),
#     observation_fc_layer_params=None,
#     action_fc_layer_params=None,
#     joint_fc_layer_params=critic_joint_fc_layer_params)
# actor_net = ActorNetwork(observation_spec, 
#                       action_spec,
#                       preprocessing_layers=preprocessing_layers,
#                       preprocessing_combiner=preprocessing_combiner)
#time_steptt = train_env.reset()
# b= actor_net(time_steptt.observation, time_steptt.step_type)
#print(time_steptt.observation)

critic_net = CriticNetwork((observation_spec,action_spec), 
                   preprocessing_layers=preprocessing_layers,
                   preprocessing_combiner=preprocessing_combiner,
                   observation_fc_layer_params=(256,),
                   action_fc_layer_params=(256,256),
                   joint_fc_layer_params=critic_joint_fc_layer_params,
                   activation_fn=tf.nn.relu
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

actor_net = ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,),
        activation_fn=tf.keras.activations.relu)

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
    batch_size=1,
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
return_value = Value('d', 0.0)
process = Process(target=compute_avg_return, args=(return_value, eval_policy, num_eval_episodes))
process.start() 
process.join() 
# avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
returns = [return_value]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_driver.run()

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = tf_agent.train(experience)

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    #compute_avg_return(return_value, eval_env, eval_policy, num_eval_episodes)
    process = Process(target=compute_avg_return, args=(return_value, eval_policy, num_eval_episodes))
    process.start() 
    process.join() 
    print('step = {0}: Average Return = {1}'.format(step, return_value))
    returns.append(return_value)

#plot the reward
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()

''' An on-policy learner learns the value of the policy being carried out by the agent including the exploration steps."
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
train_py_env.shutdown()
