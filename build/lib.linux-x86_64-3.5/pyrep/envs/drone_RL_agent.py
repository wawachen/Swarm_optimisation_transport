from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
import numpy as np
from pyrep.envs.array_specs import ArraySpec

class Drone:

    def __init__(self, id):
        self.agent = NewQuadricopter(id,4)
        self.theta = np.random.uniform(0,360) 
        self.v = None
        # self.v_cmd = None
        self.theta_cmd = None
        self.wall_collision = None

    def wall_detection(self):
        wall_dists = np.array([np.abs(10.0-self.get_2d_pos()[1]),np.abs(10.0+self.get_2d_pos()[1]),np.abs(10.0+self.get_2d_pos()[0]),np.abs(30.0-self.get_2d_pos()[0])])
        wall_collision = wall_dists < 0.1
        return wall_collision

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.6:
            return 1
        else: 
            return 0

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])
        self.theta = np.random.uniform(0,360) 
        self.v = 0.5
        self.theta_cmd = 0.0

    def hover(self,pos):
        vels = self.agent.position_controller([pos[0],pos[1],1.7])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        vels = self.agent.velocity_controller(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action_pos(self, action, pr):

        pos = np.zeros(2)
        pos[0] = self.get_2d_pos()[0] + action[0] * 0.05*5 # 0.05 is the simulation timestep
        pos[1] = self.get_2d_pos()[1] + action[1] * 0.05*5

        # print(np.sqrt(sum((pos-self.get_2d_pos()[:])**2)))

        vels = self.agent.position_controller([pos[0],pos[1],1.7])

        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

        pr.step()
        count = 0
        while np.sqrt(sum((pos-self.get_2d_pos()[:])**2)) > 0.002:
            vels = self.agent.position_controller([pos[0],pos[1],1.7])

            self.agent.set_propller_velocity(vels[:])

            self.agent.control_propeller_thrust(1)
            self.agent.control_propeller_thrust(2)
            self.agent.control_propeller_thrust(3)
            self.agent.control_propeller_thrust(4)

            pr.step()
            count += 1
        
        # print(count)
            
        
