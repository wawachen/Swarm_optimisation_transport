"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np
from tf_agents.environments import py_environment


SCENE_FILE = join(dirname(abspath(__file__)),
                  'cooperative_transportation_uav.ttt')

EPISODES = 5
EPISODE_LENGTH = 1000

class Drone_Env(py_environment.PyEnvironment):

    def __init__(self):
        #action: velocities of four rotors
        self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=0.0, maximum=1.0, name='action')
        #observations: connecting state, pose of uav and payload
        self._observation_spec = array_spec.BoundedArraySpec(shape=(12,), dtype=np.float32, minimum=[-2.5,-2.5,0.0,-np.pi,-np.pi/2,-np.pi,-2.5,-2.5,0.0,-np.pi,-np.pi/2,-np.pi], maximum=[2.5,2.5,3.0,np.pi,np.pi/2,np.pi,2.5,2.5,3.0,np.pi,np.pi/2,np.pi], name='observation')
        self._state = [0.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0]
        self._episode_ended = False

        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = Panda()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.target.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


env = ReacherEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        reward, next_state = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
