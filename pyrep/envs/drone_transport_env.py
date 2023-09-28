from os import path
from pyrep import PyRep
from pyrep.envs.drone_bacterium_agent1 import Drone
from pyrep.objects.shape import Shape
import numpy as np
from pyrep.objects.vision_sensor import VisionSensor
import math
import math

class Drone_Env_transport_TEST:

    def __init__(self,args, rank,env_name,num_agents):

        #Settings for video recorder 
        self.args = args
        fps = 2
        self.ind = 0
        # environment parameters
        self.time_step = 0
        self.safe_distance = 0.71

        # configure spaces
        self.num_a = num_agents
        self.att = np.zeros([self.num_a,2])
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        
        self.agents = [Drone(i) for i in range(num_agents)]
        self.payload = Shape('Cylinder18')

        if self.args.scenario_name == "proposed":
            self.bacterium_spread()
        if self.args.scenario_name == "ring":
            self.ring_spread()
        if self.args.scenario_name == "imbalance":
            self.imbalance_spread()
        if self.args.scenario_name == "balance":
            self.two_handle_spread()
        
        for j in range(50):
            for i in range(num_agents):
                self.agents[i].hover()
            self.pr.step()
       
        self.pos_des = np.zeros([num_agents,3])
        self.pos_des1 = np.zeros([num_agents,3])
        self.concen_collect = np.zeros(num_agents)
        self.enterlift = 0

        for i in range(num_agents):
            p_pos = self.agents[i].agent.get_drone_position()
            # obj_h = 1.2-self.agents[i].agent.get_concentration(self.payload)
            obj_h = self.agents[i].agent.get_concentration(self.payload)
            # print(obj_h)
            self.concen_collect[i] = obj_h
            # des_h = p_pos[2] - (self.agents[i].agent.suction_cup.get_suction_position()[2]-obj_h)+0.0565+0.02+0.0019-0.01
            des_h = p_pos[2] - (self.agents[i].agent.suction_cup.get_suction_position()[2]-obj_h)+0.0565+0.02+0.0019-0.01
            self.pos_des[i,0] = p_pos[0]
            self.pos_des[i,1] = p_pos[1]
            self.pos_des[i,2] = des_h

        print("des",self.pos_des)

        self.goal_x = 30
        self.goal_y = 0.0
        # self.goal_x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        # self.goal_y = [0, 1.5, 3.0, 4.5, 3.0, 1.5, 0, 0, 0, -1.5, -3.0, -4.5, -3.0, -1.5, 0]

        self.energy_swarm = np.zeros(self.num_a)

        self.energy_iter = []
        self.force_analysis = []

        self.payload_x = []
        self.payload_y = []
        self.payload_z = []

        self.pos_z = []
        self.vel_x = []
        self.vel_y = []

        self.payload_vx = []
        self.payload_vy = []
        self.payload_vz = []

        self.payload_orientation_x = []
        self.payload_orientation_y = []
        self.payload_orientation_z = []
        self.v_des = []

        self.done = 0
     

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_camera3.ttm'))
            model_handles.append(m1)

        return model_handles

    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance <= self.safe_distance:
            return 1
        else:
            return 0

    def bacterium_spread(self):
        #These positions of the grasping formation are recorded in the previous navigation part
        pos = np.array([[-0.7339,0.6171],[-0.5398,-0.7642],[-2.2253,0.0902],[1.3724,0.7583],[0.7385,1.2347],[0.0061,-1.2779],[0.4502,-0.5863],[-1.4566,-1.3668],[1.0228, -1.3844],[-0.4064,1.3336],[1.2599,-0.6185],[-0.1967,-0.0479],[-1.5125,0.5688],[0.0962,0.7089],[-0.7437,-1.5993],[-1.7331,-0.5929],[1.7958,-0.0116],[0.7111,0.1348],[-1.0491,-0.1367],[0.7705,-2.1567]])
        for i in range(self.num_a):
            px = pos[i,:][0]
            py = pos[i,:][1]
            self.agents[i].agent.set_3d_pose([px,py,1.7,0.0,0.0,0.0])
            self.att[i,0] = px
            self.att[i,1] = py


    def ring_spread(self):
        for i in range(self.num_a):
            angle = 18*i
            px = 4.5*np.cos(np.radians(angle))
            py = 4.5*np.sin(np.radians(angle))
            self.agents[i].agent.set_3d_pose([px,py,1.7,0.0,0.0,0.0])
            self.att[i,0] = px
            self.att[i,1] = py
            
        
    def two_handle_spread(self):
        for i in range(self.num_a):
            if i>9:
                i1 = i-10
                angle = 36*i1
                px = 2*np.cos(np.radians(angle))+2.5
                py = 2*np.sin(np.radians(angle))
                self.att[i,0] = px
                self.att[i,1] = py
            else:
                angle = 36*i
                px = 2*np.cos(np.radians(angle))-2.5
                py = 2*np.sin(np.radians(angle))
                self.att[i,0] = px
                self.att[i,1] = py
            self.agents[i].agent.set_3d_pose([px,py,1.7,0.0,0.0,0.0])

    def imbalance_spread(self):
        for i in range(self.num_a):
            angle = 18*i
            px = 2*np.cos(np.radians(angle))
            py = 2*np.sin(np.radians(angle))+1.5
            self.att[i,0] = px
            self.att[i,1] = py
        
            self.agents[i].agent.set_3d_pose([px,py,1.7,0.0,0.0,0.0])
            

    def random_spread(self):
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            if i == 0:
                angle = np.random.uniform(0,360)
                r = np.random.uniform(0,4.6)
                px = r*np.cos(np.radians(angle))
                py = r*np.sin(np.radians(angle))
                self.agents[i].agent.set_3d_pose([px,py,1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_drone_position()[0]
                vy = self.agents[i].agent.get_drone_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                angle = np.random.uniform(0,360)
                r = np.random.uniform(0,4.6)
                px = r*np.cos(np.radians(angle))
                py = r*np.sin(np.radians(angle))
                vpt = [px,py]
                check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)
                while check_conditions:
                    angle = np.random.uniform(0,360)
                    r = np.random.uniform(0,4.6)
                    px = r*np.cos(np.radians(angle))
                    py = r*np.sin(np.radians(angle))
                    vpt = [px,py]
                    check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                vpts.append(vpt)
                saved_agents.append(i)

    def step(self):
        self.time_step += 1

        force_readings = np.zeros(self.num_a)
        ee = np.zeros(self.num_a)

        for i in range(self.num_a):   
            force_readings[i] = self.agents[i].agent.suction_cup._attach_point.read()[0][2]
            ee[i] = np.sum(self.agents[i].agent.energy)
        self.force_analysis.append(force_readings)
        self.energy_iter.append(ee)

        ####################################
        pos_payload = self.payload.get_position()
        ori_payload = self.payload.get_orientation()
        v_payload = self.payload.get_velocity()[0]

        self.payload_x.append(pos_payload[0])
        self.payload_y.append(pos_payload[1])
        self.payload_z.append(pos_payload[2])

        self.payload_vx.append(v_payload[0])
        self.payload_vy.append(v_payload[1])
        self.payload_vz.append(v_payload[2])

        self.payload_orientation_x.append(ori_payload[0])
        self.payload_orientation_y.append(ori_payload[1])
        self.payload_orientation_z.append(ori_payload[2])

        #############################

        connect_s = np.zeros(self.num_a)

        for i in range(self.num_a):
            detect = self.agents[i].agent.suction_cup.grasp(self.payload)
            connect_s[i] = detect

        if self.time_step > 100:
            if self.time_step < 300:
                #lift the object
                if self.enterlift == 0:
                    for i in range(self.num_a):
                        p_pos = self.agents[i].agent.get_drone_position()
                        self.pos_des1[i,0] = p_pos[0]
                        self.pos_des1[i,1] = p_pos[1]
                        self.pos_des1[i,2] = 2+self.concen_collect[i]+0.0094+0.4972
                    # print('wawa',self.pos_des1)

                # ee = np.zeros(self.num_a)
                for i in range(self.num_a):
                    # flock_vel = self.agents[i].flock_controller()
                    vels = self.agents[i].agent.position_controller1(self.pos_des1[i,:])

                    self.agents[i].agent.set_propller_velocity(vels)

                    self.agents[i].agent.control_propeller_thrust(1)
                    self.agents[i].agent.control_propeller_thrust(2)
                    self.agents[i].agent.control_propeller_thrust(3)
                    self.agents[i].agent.control_propeller_thrust(4)
                    # self.energy_swarm[i] = np.sum(self.agents[i].agent.energy)+self.energy_swarm[i]
                    # ee[i] = np.sum(self.agents[i].agent.energy)
                # self.energy_iter.append(ee)
                
                self.enterlift += 1
            else:                                
                #transport to the destination
                posx_c = np.zeros(self.num_a)
                posy_c = np.zeros(self.num_a)
                for i in range(self.num_a):
                    pos_c = self.agents[i].agent.get_drone_position()
                    posx_c[i] = pos_c[0]
                    posy_c[i] = pos_c[1]
                centroid_x = np.mean(posx_c)
                centroid_y = np.mean(posy_c)

                # if ((self.time_step-300)%60)==0 and self.time_step!=300:
                #     self.ind = self.ind+1

                # v_f = np.array([self.goal_x[self.ind], self.goal_y[self.ind]]) - np.array([centroid_x,centroid_y])
                v_f = np.array([self.goal_x, self.goal_y]) - np.array([centroid_x,centroid_y])
                d = np.sqrt((v_f**2).sum())
                vector_force = v_f/d

                vel = np.zeros(3)
                v_f1 = vector_force*(d)/self.args.kp  #50/15
                # print(v_f1)
                vel[0] = v_f1[0]
                vel[1] = v_f1[1]
                self.v_des.append(vel)

                for i in range(self.num_a):
                    self.agents[i].set_action(vel)

                # print("time{}".format(self.time_step),":",ee)

                if np.sqrt(np.sum((self.payload.get_position()[:2]-np.array([self.goal_x[-1],self.goal_y[-1]]))**2))<=1.5:
                    self.done = 1

        else:
            #drop down to pick up the load
            # ee = np.zeros(self.num_a)
            for i in range(self.num_a):
                self.agents[i].agent.suction_cup.grasp(self.payload)
            ############################

            # print("time{}".format(self.time_step),":",ee)
            ############################
            for i in range(self.num_a):
                # flock_vel = self.agents[i].flock_controller()
                vels = self.agents[i].agent.position_controller1(self.pos_des[i,:])
                self.agents[i].agent.set_propller_velocity(vels)
                self.agents[i].agent.control_propeller_thrust(1)
                self.agents[i].agent.control_propeller_thrust(2)
                self.agents[i].agent.control_propeller_thrust(3)
                self.agents[i].agent.control_propeller_thrust(4)

                # self.energy_swarm[i] = np.sum(self.agents[i].agent.energy)+self.energy_swarm[i]
                # ee[i] = np.sum(self.agents[i].agent.energy)
                
            # self.energy_iter.append(ee)
        print("time",self.time_step)
        # self.suction_cup.release()

        self.pr.step()  #Step the physics simulation

        # img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.video.write(img_rgb)
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
        

   