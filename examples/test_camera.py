from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
from pyrep.robots.end_effectors.uarm_Vacuum_Gripper import UarmVacuumGripper
import numpy as np
from multiprocessing import Process,Value
from cv2 import VideoWriter, VideoWriter_fourcc
import matplotlib.pyplot as plt
import cv2
import random
import math
from pyrep.backend import sim
import imageio
from pyrep.envs.drone_bacterium_env import Drone_Env_bacterium
from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
from pyrep.drone_controller_utility.PID import PID
import numba

SCENE_FILE = join(dirname(abspath(__file__)), 'RL_testcamera_circle.ttt')#'RL_testcamera_circle.ttt'
num_agents = 2

@numba.jit
def _GetPredefined(nl,nc,src_hsv):
    for j in range(nl):
        for i in range(nc):
            # calculate very pixel
            H = src_hsv[j,i,0]
            S = src_hsv[j,i,1]
            V = src_hsv[j,i,2]
            # print(H,S,V)
            if ((H >= 35) and (H <= 77) and (S >= 43) and (S <= 255) and (V >= 46) and (V <= 255)):
                # let the green region to be black
                src_hsv[j, i, 0] = 0
                src_hsv[j, i, 1] = 0
                src_hsv[j, i, 2] = 0
            else:
                src_hsv[j, i, 0] = 0
                src_hsv[j, i, 1] = 0
                src_hsv[j, i, 2] = 255
    
    return src_hsv

# create multiagent environment
class Drone_Env_bacterium:

    def __init__(self,env_name):
        #flocking parameters
        self.near_distance = 0.5
        self.far_distance = 3

        self.T = 0
        self.T0 = 180
        self.T_vary = 0
        self.threshold_value = 30
        self.Tm = 1
        self.alpha = 1000 #300 6000
        self.kd = 2 #default 30

        self.dP_dt = 0
        self.dC_dt = 0
        self.dP_dt_weight = 0
        self.current_concentration = 0
        self.previous_concentration = 0
        self.random_bearing = 0
        self.max_speed = 1
        self.velocity = np.zeros(4)
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_z = 0

        self.memory_capacity = 20
        self.concentration_record = np.zeros(self.memory_capacity)
        self.p_rate_record = np.zeros(self.memory_capacity)
        self.estimated_position = np.zeros(3)

        self.counter = 0
        self.queque_num = 0

        self.k_sep = 1.5
        self.k_coh = 1.2
        self.k_frict = 0.5
        self.flocking_centroid = np.zeros(2)
        self.speed_max = 1

        # set the flocking level! this is the level for flocking, must to be set
        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        self.level = 1.7
        self.pre_height = 0.0

        self.last_pos_x = 0
        self.last_pos_y = 0
        self.last_pos_z = 0

        self.object_num = 0
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.agents = [NewQuadricopter(i,4) for i in range(1)]

        self.vs_floor = VisionSensor('Vision_sensor_down') 
        # for j, agent in enumerate(self.agents):
        #     agent.agent.panoramic_camera.set_resolution([1024,512])
        # self.cam = VisionSensor('Video_recorder')
        # fourcc = VideoWriter_fourcc(*'MP42')
        # self.video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))
        self.pos = self.agents[0].get_drone_position()[:]
        # self.pos1 = self.agents[2].get_drone_position()[:]
        # self.agents[0].panoramic_camera.set_resolution([480, 240])
        #obstacle avoidance
        # initialize params for detection area
        self.max_detection_distance = 0.5 + 0.22 # 0.5 is the range and 0.22 is the radius
        self.mu = 0.05 #for velocity method in FIRAS
        self.rotrix_front = self.RotationMatrix(-np.pi/2, 0, -np.pi/2)
        self.rotrix_left = self.RotationMatrix(-np.pi/2, 0, np.pi)
        self.rotrix_right = self.RotationMatrix(-np.pi/2, 0, 0)
        self.rotrix_back = self.RotationMatrix(-np.pi/2, 0, np.pi/2)

        for i in range(300):
            self.hover()
            self.pr.step()

        depth_fl = self.vs_floor.capture_depth()
        ol = (10000-np.sum(depth_fl))*10
        ol_trans = int(ol)-9537-39
        if ol_trans<0:
            ol_trans = 0
        self.previous_concentration = ol_trans

        self.agents[0].panoramic_holder.set_color([0.7, 0.0, 0.0])

    def hover(self):
        p_pos = self.agents[0].get_drone_position()

        self.vs_floor.set_position([p_pos[0]+0.02,p_pos[1],1.4602])
        self.vs_floor.set_orientation([np.radians(180),0,np.radians(90)])

        bac_vel = self.level_pid.ComputeCorrection(self.level, p_pos[2], 0.01)

        vels = self.agents[0].velocity_controller1(0,0,bac_vel)

        self.agents[0].set_propller_velocity(vels)

        self.agents[0].control_propeller_thrust(1)
        self.agents[0].control_propeller_thrust(2)
        self.agents[0].control_propeller_thrust(3)
        self.agents[0].control_propeller_thrust(4)
 
    def step(self):
        # self.agents[0].panoramic_holder.set_color()
        p_pos = self.agents[0].get_drone_position()
        self.agents[0].panoramic_holder.set_position([p_pos[0],p_pos[1],p_pos[2]+0.1],reset_dynamics=False)
        self.agents[0].panoramic_holder.set_orientation([0,0,np.radians(90)],reset_dynamics=False)

        self.vs_floor.set_position([p_pos[0]+0.02,p_pos[1],1.4602])
        self.vs_floor.set_orientation([np.radians(180),0,np.radians(90)])

        activate = self.obstacle_detection()
        activate = 0

        if activate == 1:
            vel_obs = np.zeros(2)
            obstacle_avoidance_velocity = self.obstacle_avoidance1()
            vel_obs[0] = obstacle_avoidance_velocity[0]
            vel_obs[1] = obstacle_avoidance_velocity[1]
            vels = self.agents[0].velocity_controller1(vel_obs[0],vel_obs[1])
            print("obs",vel_obs[0],vel_obs[1])

        else:
            # image = self.agents[0].get_panoramic()
        
            # image = image[:,:,(2,1,0)]
            # # print(image.shape)  
            # cv2.imshow("Original image",image)
            # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # # print(image)
            
            # # print(image.dtype)
            # vel = self.image_process(image)
            # print(vel)
            # vels = self.agents[0].velocity_controller1(vel[0],vel[1],vel[2])

            #camera_s
            self.odometry_height = self.agents[0].get_drone_position()[2]
            
            bac_vels = self.bacterium_controller_camera()
            bac_vels[2] = self.level_pid.ComputeCorrection(self.level, self.odometry_height, 0.01)
            vels = self.agents[0].velocity_controller1(bac_vels[0],bac_vels[1],bac_vels[2])
            #camera_e
            # print(vels[2])

        # p_pos = self.agents[0].get_drone_position()
        # ori_angle = self.agents[0].panoramic_holder.get_orientation()

        # if p_pos[2] - self.pre_height < 0.1:
        #     height = self.pre_height
        # else:
        #     height = p_pos[2]

        # self.agents[0].panoramic_holder.set_position([p_pos[0],p_pos[1],height+0.1])
        # self.agents[0].panoramic_holder.set_orientation([0,0,np.radians(90)])
        # self.pre_height = height

        
        self.agents[0].set_propller_velocity(vels)
        # print(vel[0]+1,vel[1]+0)

        self.agents[0].control_propeller_thrust(1)
        self.agents[0].control_propeller_thrust(2)
        self.agents[0].control_propeller_thrust(3)
        self.agents[0].control_propeller_thrust(4)

        # vels1 = self.agents[1].position_controller([self.pos[0],self.pos[1],1.7])
        # self.agents[1].set_propller_velocity(vels1)

        # self.agents[1].control_propeller_thrust(1)
        # self.agents[1].control_propeller_thrust(2)
        # self.agents[1].control_propeller_thrust(3)
        # self.agents[1].control_propeller_thrust(4)

        # vels2 = self.agents[2].position_controller([self.pos1[0],self.pos1[1],1.7])
        # self.agents[2].set_propller_velocity(vels2)

        # self.agents[2].control_propeller_thrust(1)
        # self.agents[2].control_propeller_thrust(2)
        # self.agents[2].control_propeller_thrust(3)
        # self.agents[2].control_propeller_thrust(4)
        
        self.pr.step()  #Step the physics simulation

    def obstacle_detection(self):
        activate = False
        
        if (self.agent.left_sensor.read()[0] == -1 or self.agent.left_sensor.read()[0] == 0) and (self.agent.right_sensor.read()[0] == -1 or self.agent.right_sensor.read()[0] == 0) and (self.agent.front_sensor.read()[0] == -1 or self.agent.front_sensor.read()[0] == 0) and (self.agent.back_sensor.read()[0] == -1 or self.agent.back_sensor.read()[0] == 0):
            return activate

        v_left = []
        v_right = []
        v_front = []
        v_back = []
        #pre-process of data
        left_array = np.zeros(3)
        right_array = np.zeros(3)
        back_array = np.zeros(3)
        front_array = np.zeros(3)

        self.obs = []

        if self.agent.left_sensor.read()[0] != -1:
            left_distance = self.agent.left_sensor.read()[0]
            # print("left:",left_distance)
            #  array stores the coordinate of the proximity point with respect to the sensor reference system
            left_array[0] = self.agent.left_sensor.read()[1][0]
            left_array[1] = self.agent.left_sensor.read()[1][1]
            left_array[2] = self.agent.left_sensor.read()[1][2]
            left_array = np.dot(left_array, self.rotrix_left)
           
            v_left.append(1)
            v_left.append(left_distance)
            v_left.append(left_array[0])
            v_left.append(left_array[1])
            v_left.append(left_array[2])
            self.obs.append(v_left)
            activate = True

        if self.agent.right_sensor.read()[0] != -1:
            right_distance = self.agent.right_sensor.read()[0]
            # print("right:",right_distance)
            #  array stores the coordinate of the proximity point with respect to the sensor reference system
            right_array[0] = self.agent.right_sensor.read()[1][0]
            right_array[1] = self.agent.right_sensor.read()[1][1]
            right_array[2] = self.agent.right_sensor.read()[1][2]
            right_array = np.dot(right_array, self.rotrix_right)
            
            v_right.append(2)
            v_right.append(right_distance)
            v_right.append(right_array[0])
            v_right.append(right_array[1])
            v_right.append(right_array[2])
            self.obs.append(v_right)
            activate = True

        if self.agent.front_sensor.read()[0] != -1:
            front_distance = self.agent.front_sensor.read()[0]
            # print("front:",front_distance)
            #  array stores the coordinate of the proximity point with respect to the sensor reference system
            front_array[0] = self.agent.front_sensor.read()[1][0]
            front_array[1] = self.agent.front_sensor.read()[1][1]
            front_array[2] = self.agent.front_sensor.read()[1][2]
            front_array = np.dot(front_array, self.rotrix_front)
            
            v_front.append(3)
            v_front.append(front_distance)
            v_front.append(front_array[0])
            v_front.append(front_array[1])
            v_front.append(front_array[2])
            self.obs.append(v_front)
            activate = True
        
        if self.agent.back_sensor.read()[0] != -1:
            back_distance = self.agent.back_sensor.read()[0]
            # print("back",back_distance)
            #  array stores the coordinate of the proximity point with respect to the sensor reference system
            back_array[0] = self.agent.back_sensor.read()[1][0]
            back_array[1] = self.agent.back_sensor.read()[1][1]
            back_array[2] = self.agent.back_sensor.read()[1][2]
            back_array = np.dot(back_array, self.rotrix_back)

            v_back.append(4)
            v_back.append(back_distance)
            v_back.append(back_array[0])
            v_back.append(back_array[1])
            v_back.append(back_array[2])
            self.obs.append(v_back)
            activate = True
        
        return activate

    def obstacle_avoidance1(self):
        self.odometry_height1 = self.agent.get_drone_position()[2]

        v_back_velocity = []
        v_front_velocity = []
        v_left_velocity = []
        v_right_velocity = []

        D = 0.7
        safety_distance = 0.5
        B = 3.5

        potential_field_velocity = []
        merge_v = np.zeros(3)
        pos_c = self.agent.get_drone_position()[:2]
        # print("wawa")
        for i in range(len(self.obs)):
            speed = B * (((D-1.2*self.obs[i][1])/(D-safety_distance))**2+0.1) * (pos_c-(pos_c+np.array([self.obs[i][2],self.obs[i][3]]))) / self.obs[i][1]
            # print((pos_c-(pos_c+np.array([self.obs[i][2],self.obs[i][3]]))) / self.obs[i][1])
            potential_field_velocity.append(speed)
             
        # merge
        # reset merge_velocity
        for i in range(len(potential_field_velocity)):
            merge_v[0] = merge_v[0] + potential_field_velocity[i][0] 
            merge_v[1] = merge_v[1] + potential_field_velocity[i][1] 
        
        merge_v[0] = merge_v[0]/len(potential_field_velocity)
        merge_v[1] = merge_v[1]/len(potential_field_velocity)
        merge_v[2] = self.level_pid1.ComputeCorrection(self.level, self.odometry_height1, 0.01)

        if merge_v[0]>2*self.speed_max:
            merge_v[0] = 2*self.speed_max
        elif merge_v[0]<-2*self.speed_max:
             merge_v[0] = -2*self.speed_max
            # print(v_mig[0])
        
        if merge_v[1]>2*self.speed_max:
            merge_v[1] = 2*self.speed_max
        elif merge_v[1]<-2*self.speed_max:
            merge_v[1] = -2*self.speed_max

        return merge_v

    def get_distance(self,a,b):
        return np.sqrt(((a-b)**2).sum()) 

    def obstacle_avoidance(self):
        v_back_velocity = []
        v_front_velocity = []
        v_left_velocity = []
        v_right_velocity = []

        potential_field_velocity = []
        merge_v = np.zeros(4)

        for i in range(len(self.obs)):
            if self.obs[i][0]==1:
                speed_left = self.mu * (1/self.obs[i][1] - 1/self.max_detection_distance) * 1 / (self.obs[i][1] * self.obs[i][1]) * self.left_distance_rate
                v_left_velocity.append(1)
                v_left_velocity.append(speed_left)
                v_left_velocity.append(-self.obs[i][2])
                v_left_velocity.append(-self.obs[i][3])
                v_left_velocity.append(-self.obs[i][4])
                potential_field_velocity.append(v_left_velocity)
            
            if self.obs[i][0]==2:
                speed_right = self.mu * (1/self.obs[i][1] - 1/self.max_detection_distance) * 1 / (self.obs[i][1] * self.obs[i][1]) * self.right_distance_rate
           
                v_right_velocity.append(1)
                v_right_velocity.append(speed_right)
                v_right_velocity.append(-self.obs[i][2])
                v_right_velocity.append(-self.obs[i][3])
                v_right_velocity.append(-self.obs[i][4])
                potential_field_velocity.append(v_right_velocity)

            if self.obs[i][0]==3:
                speed_front = self.mu * (1/self.obs[i][1] - 1/self.max_detection_distance) * 1 / (self.obs[i][1] * self.obs[i][1]) * self.front_distance_rate
            
                v_front_velocity.append(1)
                v_front_velocity.append(speed_front)
                v_front_velocity.append(-self.obs[i][2])
                v_front_velocity.append(-self.obs[i][3])
                v_front_velocity.append(-self.obs[i][4])
                potential_field_velocity.append(v_front_velocity)

            if self.obs[i][0]==4:
                speed_back = self.mu * (1/self.obs[i][1] - 1/self.max_detection_distance) * 1 / (self.obs[i][1] * self.obs[i][1]) * self.back_distance_rate
            
                v_back_velocity.append(1)
                v_back_velocity.append(speed_back)
                v_back_velocity.append(-self.obs[i][2])
                v_back_velocity.append(-self.obs[i][3])
                v_back_velocity.append(-self.obs[i][4])
                potential_field_velocity.append(v_back_velocity)

        # merge
        # reset merge_velocity
        merge_v[0] = 0
        merge_v[1] = 0
        merge_v[2] = 0

        for i in range(len(potential_field_velocity)):
            array = self.NormalizeVector(potential_field_velocity[i][2], potential_field_velocity[i][3], potential_field_velocity[i][4])
            merge_v[0] = merge_v[0] + potential_field_velocity[i][1] * array[0]
            merge_v[1] = merge_v[1] + potential_field_velocity[i][1] * array[1]
            merge_v[2] = merge_v[2] + potential_field_velocity[i][1] * array[2]

        merge_speed = np.sqrt(merge_v[0]*merge_v[0] + merge_v[1]*merge_v[1] + merge_v[2]*merge_v[2])
        merge_v[3] = merge_speed

        return merge_v

    def bacterium_controller_camera(self):
        
        # if nearest_dis < 2.0:
        #     obs = 1
        # else:
        #     obs = 0
        
        self.counter += 1

        depth_fl = self.vs_floor.capture_depth()
        cv2.imshow("depth image",depth_fl)


        # if np.sum(depth_fl) == 0:
        #     return
        # print(np.sum(depth_fl))

        ol = (10000-np.sum(depth_fl))*10
        ol_trans = int(ol)-9537-39-10959
        if ol_trans<0:
            ol_trans = 0

        concen = ol_trans
        
        self.current_concentration = concen
        print(self.current_concentration)
        self.dC_dt = self.current_concentration - self.previous_concentration

        # if self.dC_dt >-100 and self.dC_dt<100:
        #     self.dC_dt = 0

        print("diff",self.dC_dt)
        # print(self.current_concentration)
        self.dP_dt = self.kd/((self.kd + self.current_concentration)*(self.kd + self.current_concentration)) * (self.dC_dt*10000)
        
        if self.queque_num < self.memory_capacity:
            self.p_rate_record[self.queque_num] = self.dP_dt
        else:
            for i in range(self.memory_capacity):
                if i == (self.memory_capacity-1):
                    self.p_rate_record[i] = self.dP_dt
                else:
                    self.p_rate_record[i] = self.p_rate_record[i+1]
        
        for i in range(self.memory_capacity):
            self.dP_dt_weight = self.dP_dt_weight + self.p_rate_record[i] * np.exp((i-19)/self.Tm)

        print("uu",self.p_rate_record)
        # print("concen",self.current_concentration)
        if self.current_concentration == 0:
            self.speed = self.max_speed
        elif self.current_concentration>0:
            self.speed = 2000*self.max_speed/(self.current_concentration)
        
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        
        # print(self.speed)
        
        self.dP_dt_weight = self.dP_dt_weight/self.Tm
        print("p_t",self.dP_dt_weight)
        if self.current_concentration == 0:
            self.T_vary += 1
            if self.T_vary > 100:
                self.T_vary = 100
        else:
            self.T_vary = 0

        self.T = (self.T0+self.T_vary) * np.exp(self.alpha * self.dP_dt_weight)
        
        print("T",self.T)
        # print(self.alpha * self.dP_dt_weight)
        
        self.dP_dt_weight = 0

        # if (self.dC_dt < 0.0008 and self.dC_dt>0) or (self.dC_dt >-0.0008 and self.dC_dt<0): #to avoid small swing
        #     self.dC_dt = 0 

        self.previous_concentration = self.current_concentration

        # if obs == 1:
        #     self.obstacleVelocityGenerator(closest_wall)
        #     self.counter = 0
        # else: 
        if self.counter > self.T:
            self.RandomVelocityGenerator1()       
            self.counter = 0
        else:
            self.velocity[3] = self.speed
            self.velocity[0] = self.speed * np.cos(self.random_bearing)
            self.velocity[1] = self.speed * np.sin(self.random_bearing)
                # print('wawa')

        self.queque_num += 1

        # print(self.velocity)
        
        return self.velocity

    def RandomVelocityGenerator1(self):
        random_seed = np.random.randint(1000)
        np.random.seed(random_seed)

        angle = (59.0 / 1.0) + (np.random.random() * 9.0)
        angle = angle + self.random_bearing

        if angle > 360:
            angle = angle - 360

        #  get a random bearing for the agent and bearing belongs to [0,360]
        # self.random_bearing = np.random.uniform(0,360) #(59.0 / 1.0) + (np.random.random() * 9.0) 
        # convert into radian
        self.random_bearing = angle
        self.random_bearing_r = np.radians(self.random_bearing)
        # generate velocity cmd
        self.velocity_x = self.speed * np.cos(self.random_bearing_r)
        self.velocity_y = self.speed * np.sin(self.random_bearing_r)

        # if self.position[0]<self.field_x_min+1 or self.position[0]>self.field_x_max-1:
        #     self.velocity_x = - self.max_speed * self.position[0]/np.sqrt(self.position[0]*self.position[0] + self.position[1]*self.position[1])
    
        # if self.position[1]<self.field_y_min+1 or self.position[1]>self.field_y_max-1:
        #     self.velocity_y = - self.max_speed * self.position[1]/np.sqrt(self.position[0]*self.position[0] + self.position[1]*self.position[1])
        
        self.velocity[0] = self.velocity_x
        self.velocity[1] = self.velocity_y
        self.velocity[2] = self.velocity_z
        self.velocity[3] = self.speed


    def RotationMatrix(self,gamma, theta, alpha):

        #  calculate the rotation matrix
        x_rotation_matrix = np.array([[1,0,0],[0,math.cos(gamma),-math.sin(gamma)],[0, math.sin(gamma), math.cos(gamma)]])
        y_rotation_matrix = np.array([[math.cos(theta), 0, -math.sin(theta)],[0,1,0],[math.sin(theta), 0, math.cos(theta)]])
        z_rotation_matrix = np.array([[math.cos(alpha), -math.sin(alpha), 0],[math.sin(alpha), math.cos(alpha), 0],[0,0,1]])

        rotation_matrix = np.dot(x_rotation_matrix, y_rotation_matrix)
        rotation_matrix = np.dot(rotation_matrix, z_rotation_matrix)

        return rotation_matrix


    def command_generate(self,pos):
        self.odometry_height = self.agents[0].get_drone_position()[2]
        
        #add the friction between drones
        dt = 0.05
        #how many drones that can be detected in the image(but not sure whether they are in the mid area)
        object_num = len(pos)
        print("dis",pos)
        pos_xyz = np.zeros([object_num,3])
        
        # initialize the vectors
        v_coh = np.zeros(3)
        v_sep = np.zeros(3)
        v_mig = np.zeros(3)
        
        # sep_sum.clear()
        sep_sum = np.zeros(3)
        
        # coh_sum.clear()
        coh_sum = np.zeros(3)

        # assumption_pos.clear()
        assumption_pos = np.zeros(3)

        # relative_velocity.clear()
        relative_velocity = np.zeros(3)

        # v_frict.clear()
        v_frict = np.zeros(3)

        # initialize the centroid of the flocking
        self.flocking_centroid[0] = 0
        self.flocking_centroid[1] = 0
        o_count = 0
        # get every vector, from the pole coordinates to xyz coordinates
        for i in range(object_num):
            theta = pos[i][0]
            phi = pos[i][1]
            distance = pos[i][2]

            # get the coordinates of the objects in initial frame
            pos_xyz[i][0] = distance * math.sin(theta) * math.cos(phi)
            pos_xyz[i][1] = -distance * math.cos(theta) * math.cos(phi)

            # iteration for calculate the centroid of the flocking
            self.flocking_centroid[0] = self.flocking_centroid[0] + pos_xyz[i][0]
            self.flocking_centroid[1] = self.flocking_centroid[1] + pos_xyz[i][1]

            #minus 0.1 because the distance between green ball and drone is 0.1
            pos_xyz[i][2] = distance * math.sin(phi) - 0.1

            print("wawa",pos_xyz[i])

            # determine the effection area and set the angula
            if distance <= self.far_distance and distance >= self.near_distance and abs(phi) <= 30*np.pi/180:
                o_count+=1
                # separation speed
                # ensure the value is not 0, which may cause nan problem
                if np.sqrt(self.Dot(pos_xyz[i], pos_xyz[i])) != 0:
                    sep_sum[0] = sep_sum[0] + pos_xyz[i][0] / np.sqrt(self.Dot(pos_xyz[i], pos_xyz[i]))
                    sep_sum[1] = sep_sum[1] + pos_xyz[i][1] / np.sqrt(self.Dot(pos_xyz[i], pos_xyz[i]))
                    sep_sum[2] = sep_sum[2] + pos_xyz[i][2] / np.sqrt(self.Dot(pos_xyz[i], pos_xyz[i]))

                # aggregation speed
                coh_sum[0] = coh_sum[0] + pos_xyz[i][0]
                coh_sum[1] = coh_sum[1] + pos_xyz[i][1]
                coh_sum[2] = coh_sum[2] + pos_xyz[i][2]

                # assume the other drones as one drone in order to calculate the velocity in total
                assumption_pos[0] = assumption_pos[0] + pos_xyz[i][0]
                assumption_pos[1] = assumption_pos[1] + pos_xyz[i][1]
                assumption_pos[2] = assumption_pos[2] + pos_xyz[i][2]
            
        #calculate the centroid of the flocking, this coordinate is under the body reference system
        self.flocking_centroid[0] = self.flocking_centroid[0]/(object_num+1)
        self.flocking_centroid[1] = self.flocking_centroid[1]/(object_num+1)

        if o_count > 0:
            # calculate the assumption drone relative velocity and friction
            relative_velocity[0] = (assumption_pos[0] - self.last_pos_x)/dt
            relative_velocity[1] = (assumption_pos[1] - self.last_pos_y)/dt
            relative_velocity[2] = (assumption_pos[2] - self.last_pos_z)/dt

            v_frict[0] =  self.k_frict * relative_velocity[0]
            v_frict[1] =  self.k_frict * relative_velocity[1]
            v_frict[2] =  self.k_frict * relative_velocity[2]

            # record the last pos and time
            self.last_pos_x = assumption_pos[0]
            self.last_pos_y = assumption_pos[1]
            self.last_pos_z = assumption_pos[2]
        else:
            v_frict[0] = 0
            v_frict[1] = 0
            v_frict[2] = 0
        
        # velocity of separation
        if object_num == 0:
            v_sep[0] = 0
            v_sep[1] = 0
            v_sep[2] = 0

            # velocity of cohision
            v_coh[0] = 0
            v_coh[1] = 0
            v_coh[2] = 0
        else:
            v_sep[0] = - self.k_sep / object_num * sep_sum[0]
            v_sep[1] = - self.k_sep / object_num * sep_sum[1]
            v_sep[2] = - self.k_sep / object_num * sep_sum[2]

            # velocity of cohision
            v_coh[0] = self.k_coh / object_num * coh_sum[0]
            v_coh[1] = self.k_coh / object_num * coh_sum[1]
            v_coh[2] = self.k_coh / object_num * coh_sum[2]

        # # velocity of migration for leader follower model
        # # onlt use the cohesion on verticle level when using learder-follower model
        # v_mig[0] = v_coh[0] + v_sep[0]
        # v_mig[1] = v_coh[1] + v_sep[1]
        # v_mig[2] = v_coh[2]

        # print(v_sep[0])
        # velocity of migration for decentralized model
        v_mig[0] = v_coh[0] + v_sep[0] + v_frict[0]
        v_mig[1] = v_coh[1] + v_sep[1] + v_frict[1]
        print("v_mig",v_mig[0],"sep",v_sep[0],"coh",v_coh[0],"frict",v_frict[0])
        # set the maximum speed for flocking
        if v_mig[0]>0.7071*self.speed_max:
            v_mig[0] = 0.7071*self.speed_max
        
        if v_mig[1]>0.7071*self.speed_max:
            v_mig[1] = 0.7071*self.speed_max

        # flocking_speed = sqrt(v_mig[0]*v_mig[0] + v_mig[1]*v_mig[1])
        #  add a height controller to set the drones on the same level in the decentral model!
        v_mig[2] = self.level_pid.ComputeCorrection(self.level, self.odometry_height, 0.01)

        # self.v_img = v_mig

        return v_mig


    def image_process(self, image):
        # cv2.imshow("Original image",image)
        panoramic_img = cv2.flip(image, -1) 
        # print(panoramic_img.shape)
        # get the image resolution size
        self.resolution_x = image.shape[1]
        self.resolution_y = image.shape[0]

        panoramic_img = self.GetPredefinedArea(panoramic_img)
        # print(panoramic_img.shape)
        
        # this calculate the area, center point coordinates, and distance and bearing
        vel = self.ConnectedComponentStatsDetect(panoramic_img)

        return vel


    def GetPredefinedArea(self, src):
        # cv2.imshow("Original image",src)
        #convert image into HSV type because this type is similar to the human eyes
        src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) 
        # print(src_hsv.dtype)
        # cv2.imshow("hsv image",src_hsv)
        # Histogram equalisation on the V-channel
        
        # src.dtype = np.uint8
        # print(src.shape)
        # cv2.imshow("Original image",src)
        src_hsv[:, :, 2] = cv2.equalizeHist(src_hsv[:, :, 2])
        
        # cv2.imshow("hsv image",src_hsv)
        # cv2.imshow("equal image",src)
        #equalize the value channal to make sure the image has the suitable contrast
        # hsvSplit.resize(3);
        # (B, G, R) = cv2.split(src) #split(src, hsvSplit);
        # # B = cv2.equalizeHist(B) #equalizeHist(hsvSplit[2],hsvSplit[2]);
        # src = cv2.merge([B, G, R])#merge(hsvSplit, src);

        nl = src_hsv.shape[0]
        nc = src_hsv.shape[1]

        src_hsv = _GetPredefined(nl,nc,src_hsv)

        # cv2.imshow("Original image",src_hsv)
        # print(src.shape)
        src2 = cv2.cvtColor(src_hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("Original image",src2)
        
        return src2
    
    def ConnectedComponentStatsDetect(self,image):
        # print(image.shape)
        pos = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("final image",gray)
        # print(gray.shape)
        # gray.dtype = np.uint8
        binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # cv2.imshow("final image",binary)
        binary = cv2.bitwise_not(binary)
        output = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_16U)

        num_labels= output[0]
        # print(num_labels)
        labels= output[1]
        stats= output[2]
        centroids= output[3]
        # print(centroids)

        w = image.shape[1]
        h = image.shape[0]

        for i in range(1,num_labels):
            v = []
            pt = np.uint8(centroids[i,:])
            pt1 = centroids[i,:]
            # print(pt)
            x = stats[i,0]
            y = stats[i,1]
            width = stats[i, 2]
            height = stats[i, 3]
            area = stats[i, 4]

            # print(pt,x,y,width,height,area)
            panoramic_img = cv2.circle(image, (pt[0], pt[1]), 2, (0, 0, 255),  -1)
            panoramic_img = cv2.rectangle(panoramic_img, (x, y), (x+width,y+height), (255, 0, 255), 1) 
            # cv2.imshow("final image",panoramic_img)
            theta = (-pt1[0] + 3*self.resolution_x/4)/self.resolution_x *2* np.pi
            y_ = self.resolution_y/2 - pt1[1]+0.5
            phi = y_ * 0.35 / 180 * np.pi

            a = 370
            b = -0.6
            c = 100
            d = 1.863
            area = area * (a+c)/(a*np.exp(b*phi)+c*np.exp(d*phi))

            # y = a*x^b, where a = 467, b = -2
            distance = np.sqrt(0.25*470/area)

            #  the first value is bearing, second is distance
            v.append(theta)
            v.append(phi)
            v.append(distance)
            # print(theta,phi,distance)
            pos.append(v)
            # vector<double>().swap(v);

        vel = self.command_generate(pos) #pass the params by const reference

        return vel
        
    def Dot(self, v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        return dot
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

   

env = Drone_Env_bacterium(SCENE_FILE)

try:
    while True:
        env.step()
finally:
    env.shutdown()