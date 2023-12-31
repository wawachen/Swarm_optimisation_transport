from os import path
from pyrep import PyRep
from pyrep.envs.drone_bacterium_agent import Drone

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
import random
import math


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env_bacterium:

    def __init__(self,args,env_name,num_agents):

        # environment parameters
        self.time_step = 0
        self.args = args

        # configure spaces
        self.num_a = num_agents

        self.safe_distance = 0.71
        self.x_limit_min = -5+self.safe_distance/2
        self.x_limit_max = 5-self.safe_distance/2
        self.y_limit_min = -5+self.safe_distance/2
        self.y_limit_max = 5-self.safe_distance/2
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        # self.payload = Shape('Cylinder18')
        # self.pr.remove_model(self.payload)

        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(num_agents)]
        self.payload = Shape('Cuboid4')

        self.goals = self.generate_goal()

        # for j, agent in enumerate(self.agents):
        #     agent.agent.panoramic_camera.set_resolution([1024,512])

        if not args.evaluate:
            self.flock_spread()
        else:
            self.direct_spread()

        self.done = 0
        # self.flock_spread()

        # self.cam = VisionSensor('Video_recorder')
        # fourcc = VideoWriter_fourcc(*'MP42')
        # self.video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))

        #For numerical analysis
        self.pos_x = []
        self.pos_y = []
        self.pos_z = []

        self.vel_x = []
        self.vel_y = []
        self.vel_z = []

        self.T_collection = []
        self.d_coll = []
        self.cen_coll = []
        self.c_sum = []

        self.attach_points = np.zeros([self.num_a,3])
        self.fail_num = 0
        self.first_enter = np.zeros(self.num_a)
        self.first_counter = np.zeros(self.num_a)
        self.old_centroid = np.zeros(2)

        for i in range(num_agents):
            self.agents[i].agent.panoramic_holder.set_color([1.0, 0.0, 0.0]) 

        for j in range(50):
            for i in range(num_agents):
                self.agents[i].hover()
            self.pr.step()

        for i in range(num_agents):
            self.agents[i].previous_concentration = self.agents[i].agent.get_concentration_camera()

        self.agent_states = np.zeros(self.num_a)
        self.high_states = np.zeros(self.num_a)
        # self.hover_states = np.zeros(self.num_a)

    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        # if not self.close_simulation:
        #     for i in range(self.num_a):
        #         self.pr.remove_model(self.model_handles[i])
        #     self.pr.remove_model(self.load_handle)
        
        self.flock_spread()
        self.done = 0
        # self.flock_spread()

        # self.cam = VisionSensor('Video_recorder')
        # fourcc = VideoWriter_fourcc(*'MP42')
        # self.video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))

        #For numerical analysis
        self.pos_x = []
        self.pos_y = []
        self.pos_z = []

        self.vel_x = []
        self.vel_y = []
        self.vel_z = []

        self.T_collection = []
        self.d_coll = []
        self.cen_coll = []
        self.c_sum = []

        self.attach_points = np.zeros([self.num_a,3])
        self.fail_num = 0
        self.first_enter = np.zeros(self.num_a)
        self.first_counter = np.zeros(self.num_a)
        self.old_centroid = np.zeros(2)

        for i in range(self.num_a):
            self.agents[i].agent.panoramic_holder.set_color([1.0, 0.0, 0.0]) 

        for j in range(50):
            for i in range(self.num_a):
                self.agents[i].hover()
            self.pr.step()

        for i in range(self.num_a):
            self.agents[i].previous_concentration = self.agents[i].agent.get_concentration_camera()

        self.agent_states = np.zeros(self.num_a)
        self.high_states = np.zeros(self.num_a)


    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_camera3.ttm'))
            model_handles.append(m1)
        
        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))
        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))

        return model_handles

    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance <= self.safe_distance:
            return 1
        else:
            return 0
    
    # def check_inside(self,pos):
    #     if pos[0] > -7 and pos[0] < 7 and pos[1] > -7 and pos[1] < 7:
    #         return 1
    #     else:
    #         return 0

    def flock_spread(self):
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
    
    def direct_spread(self):

        if self.num_a == 3:
            pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)
        
        if self.num_a == 4:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,-4.5],[-3.5,4.5]])
            # pos = np.array([[1.5,4.5],[3.5,4.5],[-1.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            pos = np.array([[4.5,4.5],[4.5,-4.5],[-4.5,4.5],[-4.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,4.5],[-3.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],1.7,0.0,0.0,0.0])


    # def random_position_spread(self):
    #     saved_agents = []
    #     vpts = []
    #     for i in range(self.num_a):
    #         if i == 0:
    #             self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.7,0.0,0.0,0.0])
    #             vx = self.agents[i].agent.get_drone_position()[0]
    #             vy = self.agents[i].agent.get_drone_position()[1]
    #             vpts.append([vx,vy])
    #             saved_agents.append(i)
    #         else:
    #             vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
    #             check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
    #             check_conditions = np.sum(check_list)
    #             while check_conditions or self.check_inside(vpt):
    #                 vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
    #                 check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
    #                 check_conditions = np.sum(check_list)

    #             self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
    #             vpts.append(vpt)
    #             saved_agents.append(i)
    
    def evaluate_trigger(self):
        cx = np.zeros(self.num_a)
        cy = np.zeros(self.num_a)

        for i in range(self.num_a):
            pos = self.agents[i].agent.get_drone_position()
            cx[i] = pos[0]
            cy[i] = pos[1]

        centroid_x = np.mean(cx)
        centroid_y = np.mean(cy)

        centroid = np.array([centroid_x,centroid_y])
        d = np.sqrt((centroid**2).sum())

        diff = np.sqrt(((centroid - self.old_centroid)**2).sum())
        # print("centroid diff", diff)
        # print("centroid", d)
        self.d_coll.append(d)
        self.cen_coll.append(diff)

        self.old_centroid = centroid

        if d < 0.4 and diff < 0.001:
            signal = 1
        else:
            signal = 0
        
        return signal

    def check_in_peanut(self, pos):
        if pos[0] < 0:
            lc = 1
            rc = 0
        else:
            rc = 1
            lc = 0

        return lc,rc

    def evaluate_trigger1(self):
        cx = []
        cy = []

        cx1 = []
        cy1 = []

        rc = 0
        lc = 0

        for i in range(len(self.agents)):
            pos = self.agents[i].agent.get_drone_position()
            lcc,rcc = self.check_in_peanut(pos)

            if lcc:
                cx.append(pos[0]) 
                cy.append(pos[1])
            if rcc:
                cx1.append(pos[0])
                cy1.append(pos[1])
        
        # assert((rc+lc)==len(self.agents))

        centroid_xl = np.mean(cx)
        centroid_yl = np.mean(cy)

        centroid_xr = np.mean(cx1)
        centroid_yr = np.mean(cy1)

        cen_compare_l = np.array([-3,0]) 
        cen_compare_r = np.array([3,0])

        centroidl = np.array([centroid_xl,centroid_yl])
        centroidr = np.array([centroid_xr,centroid_yr])

        dl = np.sqrt(((centroidl-cen_compare_l)**2).sum())
        dr = np.sqrt(((centroidr-cen_compare_r)**2).sum())
        d = dl+dr

        self.d_coll.append(d)

        if d < 0.07 and (np.sum(self.high_states)==self.num_a):
            signal = 1 #(np.sum(check_inside)==len(self.agents))
        else:
            signal = 0

        return signal

    def generate_goal(self):
        #####################################
        #visualization goal
        if self.args.load_type == "three":
            self.payload_1 = Shape('Cuboid28')
            self.payload_2 = Shape('Cuboid29')
            self.payload_3 = Shape('Cuboid30')

            loads = [self.payload_1,self.payload_2,self.payload_3]

        if self.args.load_type == "four":
            self.payload_1 = Shape('Cuboid24')
            self.payload_2 = Shape('Cuboid25')
            self.payload_3 = Shape('Cuboid28')
            self.payload_4 = Shape('Cuboid29')

            loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4]
        
        if self.args.load_type == "six":
            self.payload_1 = Shape('Cuboid24')
            self.payload_2 = Shape('Cuboid25')
            self.payload_3 = Shape('Cuboid26')
            self.payload_4 = Shape('Cuboid27')
            self.payload_5 = Shape('Cuboid28')
            self.payload_6 = Shape('Cuboid29')

            loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4,self.payload_5,self.payload_6] 

        points = []

        for i in range(len(loads)):
            points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            
        goal_points = np.array(points)

        return goal_points

    def reward_and_terminate(self):
        rew = 0
        done_terminate = 0
        finish_sig = np.zeros(self.num_a)

        #team reward
        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.5))
            rew -= min(dists)/(5*2)
            
        if np.all(finish_sig):
            done_terminate = 1 

        return rew,done_terminate

    def step(self):
        print("time",self.time_step)
        self.time_step += 1
        x_p = np.zeros([self.num_a,1])
        y_p = np.zeros([self.num_a,1])
        z_p = np.zeros([self.num_a,1])

        x_v = np.zeros([self.num_a,1])
        y_v = np.zeros([self.num_a,1])
        z_v = np.zeros([self.num_a,1])
        t_c = np.zeros([self.num_a,1])

        signal_t = self.evaluate_trigger1()

        for i in range(self.num_a):
            if self.agents[i].agent.get_drone_position()[2]<1.6:
                self.agent_states[i] = 1
            else:
                self.agent_states[i] = 0

        max_concen = np.zeros(self.num_a)
        for i in range(self.num_a):
            if self.agent_states[i] == 0:
                max_concen[i] = self.agents[i].agent.get_concentration_camera()
            else:
                max_concen[i] = 0 

        self.c_sum.append(np.sum(max_concen))    
        flock_concen = np.max(max_concen)
        print('max',flock_concen)

        for i in range(self.num_a):
            if max_concen[i]>(flock_concen*0.70):
                self.agents[i].agent.panoramic_holder.set_color([0.0, 1.0, 0.0])
                self.high_states[i] = 1
            else:
                self.agents[i].agent.panoramic_holder.set_color([1.0, 0.0, 0.0])
                self.high_states[i] = 0

        for i,agent in enumerate(self.agents):
            p_i = agent.agent.get_drone_position()
            
            v_i = agent.agent.get_velocities()[0]
           
            self.attach_points[i,0] = p_i[0]
            self.attach_points[i,1] = p_i[1]
            self.attach_points[i,2] = p_i[2]
            
            x_p[i,0] = p_i[0]
            y_p[i,0] = p_i[1]
            z_p[i,0] = p_i[2]
            # print("wawa")
            x_v[i,0] = v_i[0]
            y_v[i,0] = v_i[1]
            z_v[i,0] = v_i[2]

            #test
            if agent.agent.get_drone_position()[2]<1.3:
                self.agent_states[i] = 1
                self.first_enter[i] = 1
                print("rescue!")
                self.pr.remove_model(self.model_handles[i])
                robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_camera3.ttm'))
                self.model_handles[i] = m1
                self.agents[i] = Drone(i)

                vpt = np.random.uniform(3,4.8,2)

                check_list = [self.check_collision(vpt,self.agents[i1].agent.get_drone_position()[:2]) if i1!= i else 0 for i1 in range(self.num_a)]
                check_conditions = np.sum(check_list)
                while check_conditions:
                    vpt = np.random.uniform(3,4.8,2)
                    check_list = [self.check_collision(vpt,self.agents[i1].agent.get_drone_position()[:2]) if i1!= i else 0 for i1 in range(self.num_a)]
                    check_conditions = np.sum(check_list)

                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                self.agents[i].agent.drone_reset()
                self.fail_num += 1

                # self.agents[i].hover()
            
            elif self.first_enter[i] == 1:
                self.agent_states[i] = 1
                self.first_counter[i] +=1
                if self.first_counter[i] == 50:
                    self.agents[i].previous_concentration = self.agents[i].agent.get_concentration_camera()
                    self.first_enter[i] = 0 
                    self.first_counter[i] = 0
                else:
                    self.agents[i].hover()

            else: 
                self.agent_states[i] = 0
                vel = np.zeros(3)

                p_pos = agent.agent.get_drone_position()
                wall_dists = np.array([np.abs(10.0-p_pos[1]),np.abs(10.0+p_pos[1]),np.abs(10.0+p_pos[0]),np.abs(10.0-p_pos[0])])
                closest_wall = np.argmin(wall_dists)
                nearest_dis = wall_dists[closest_wall]

                agent.agent.depth_sensor.set_position([p_pos[0],p_pos[1],p_pos[2]-0.5])
                agent.agent.depth_sensor.set_orientation([-np.radians(180),0,0])

                agent.agent.vs_floor.set_position([p_pos[0]+0.02,p_pos[1],1.4602])
                agent.agent.vs_floor.set_orientation([np.radians(180),0,np.radians(90)])

                agent.agent.panoramic_holder.set_position([p_pos[0],p_pos[1],p_pos[2]+0.1],reset_dynamics=False)
                agent.agent.panoramic_holder.set_orientation([0,0,np.radians(90)],reset_dynamics=False)

                agent.agent.proximity_holder.set_position([p_pos[0],p_pos[1],p_pos[2]+0.1],reset_dynamics=False)
                agent.agent.proximity_holder.set_orientation([0,0,0],reset_dynamics=False)
                # vel = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[:2]
        
                activate = agent.obstacle_detection()
                # activate = 0

                if activate == 1:
                    if agent.agent.get_concentration_camera() > 0:
                        # q_o = 1.2-agent.agent.get_concentration()
                        q_o = agent.agent.get_concentration_camera()
                        vel_obs = np.zeros(3)
                        obstacle_avoidance_velocity = agent.obstacle_avoidance1()
                        vel_obs[0] = obstacle_avoidance_velocity[0]/(q_o/3000)
                        vel_obs[1] = obstacle_avoidance_velocity[1]/(q_o/3000)
                        # vel_obs[0] = 0.3*agent.flock_controller()[0] + obstacle_avoidance_velocity[0]/(q_o*100)+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                        # vel_obs[1] = 0.3*agent.flock_controller()[1] + obstacle_avoidance_velocity[1]/(q_o*100)+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                        vel_obs[2] = obstacle_avoidance_velocity[2]

                        agent.set_action(vel_obs)
                        # print("obs_internal",i,vel_obs[0],vel_obs[1])
                    else:
                        vel_obs = np.zeros(3)
                        obstacle_avoidance_velocity = agent.obstacle_avoidance1()
                        vel_obs[0] = obstacle_avoidance_velocity[0]#+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]
                        vel_obs[1] = obstacle_avoidance_velocity[1]#+agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                        vel_obs[2] = obstacle_avoidance_velocity[2]

                        if nearest_dis < 1:
                            # print("uuuu")
                            agent.counter = 1000000

                        agent.set_action(vel_obs)
                        # print("obs",i,vel_obs)

                else:
                    if agent.agent.get_concentration_camera() > (flock_concen*0.70):
                        
                        if agent.is_neighbour()>0:
                            # print("wawa")
                            flock_vel = agent.flock_controller()
                            bac_vel = agent.bacterium_controller_camera()
                            # shepherd_vel = agent.shepherd_controller()

                            vel[0] = 0.0*flock_vel[0] + bac_vel[0] 
                            # vel[0] = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[0]

                            vel[1] = 0.0*flock_vel[1] + bac_vel[1] 
                            # vel[1] = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[1]
                            # print("comb",bac_vel,flock_vel)
                            vel[2] = flock_vel[2]
                            agent.set_action(vel)
                            # print("bac_in",vel)
                            # print("shepherd",0.2*shepherd_vel)
                        
                        else:
                            # print("wawa1")
                            # shepherd_vel = agent.shepherd_controller()
                            bac_vel = agent.bacterium_controller_camera()
                            flock_vel = agent.flock_controller()

                            vel[0] = bac_vel[0]*2
                            vel[1] = bac_vel[1]*2
                            vel[2] = flock_vel[2]
                            # print("bacterium",vel)
                            # print("shepherd",0.2*shepherd_vel)
                            agent.set_action(vel)
                    else:
                        if agent.agent.get_concentration_camera() > (flock_concen*0.50):
                            bac_vel = agent.bacterium_controller_camera() 
                            flock_vel = agent.flock_controller()  
                            shepherd_vel = agent.shepherd_controller()
                            vel[0] = 2*bac_vel[0] + 0.2*shepherd_vel[0]#0.3*flock_vel[0] + bac_vel[0]
                            vel[1] = 2*bac_vel[1] + 0.2*shepherd_vel[1]#0.3*flock_vel[0] + bac_vel[1]
                            vel[2] = flock_vel[2] 
                            # print(bac_vel)
                            agent.set_action(vel)
                        else: 
                            bac_vel = agent.bacterium_controller_camera() 
                            flock_vel = agent.flock_controller()
                            shepherd_vel = agent.shepherd_controller()  
                            vel[0] = bac_vel[0]+0.2*shepherd_vel[0]
                            vel[1] = bac_vel[1]+0.2*shepherd_vel[1]
                            vel[2] = flock_vel[2] 
                            # print(bac_vel)
                            agent.set_action(vel)
                            # print("bac",vel)
                
                # vel = agent.flock_controller()
                # print(i,obstacle_avoidance_velocity[0]/(q_o*100),obstacle_avoidance_velocity[1]/(q_o*100))
                # print('bacterium',agent.bacterium_controller()[:2], 'flock', agent.flock_controller()[:2])
                # vel = agent.bacterium_controller(nearest_dis,closest_wall,self.payload)[:2]
                # print(agent.is_neighbour())
                # vel = agent.flock_controller()
            t_c[i,0] = agent.T    
                # agent.set_action(vel)
       
        self.pos_x.append(x_p)
        self.pos_y.append(y_p)
        self.pos_z.append(z_p)

        self.vel_x.append(x_v)
        self.vel_y.append(y_v)
        self.vel_z.append(z_v)

        self.T_collection.append(t_c)
        
        self.pr.step()  #Step the physics simulation

        if self.num_a == 3:
            if self.time_step>= 2800: #2800
                self.done = 1
        
        if self.num_a == 4:
            if self.time_step>= 3800: #2800
                self.done = 1
        
        if self.num_a == 6:
            if self.time_step>= 4800: #2800
                self.done = 1
        
        reward_n = []
        done_ter = []
        for i, agent in enumerate(self.agents):
            #----------------------------
            rw,dter = self.reward_and_terminate()
            reward_n.append(rw)
            done_ter.append(dter)
            
        #all agents get total reward in cooperative case
        reward = reward_n[0] #need modify

        if np.all(done_ter):
            reward = 1*self.num_a

        #once collision every agent will be pulished
        reward_n = [reward] * self.num_a

        return reward_n, np.all(done_ter), [x_p,y_p,z_p], [x_v,y_v,z_v]

        # img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.video.write(img_rgb)
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
        

   

