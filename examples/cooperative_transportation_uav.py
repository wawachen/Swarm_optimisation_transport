from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
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


safe_distance = 0.71
x_limit_min = -10+safe_distance/2
x_limit_max = 30-safe_distance/2
y_limit_min = -10+safe_distance/2
y_limit_max = 10-safe_distance/2

def check_collision(point,point1):
    distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
    if distance <= safe_distance:
      return 1
    else:
      return 0

fps = 24

LOOPS = 13
SCENE_FILE = join(dirname(abspath(__file__)), 'RL_uniform_nobound.ttt')

pr = PyRep()  
pr.launch(SCENE_FILE, headless=False)
pr.start()

cam = VisionSensor('Video_recorder')

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (cam.get_resolution()[0], cam.get_resolution()[1]))


robot_DIR = path.join(path.dirname(path.abspath(__file__)), 'models')


#pr.remove_model(m1)
[m,m1]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
[m_1,m1_1]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
[m_2,m1_2]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
[m_3,m1_3]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
[m_4,m1_4]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
[m_5,m1_5]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))

#pr.remove_model(m1)
#agent = Quadricopter()
# agent1 = NewQuadricopter(0,4)
# agent2 = NewQuadricopter(1,4)
# agent3 = NewQuadricopter(2,4)
# agent4 = NewQuadricopter(3,4)
# agent5 = NewQuadricopter(4,4)
# agent6 = NewQuadricopter(5,4)

#target = Shape('Payload')
#suction_cup = UarmVacuumGripper()

# We could have made this target in the scene, but lets create one dynamically
#target = Shape.create(type=PrimitiveShape.CUBOID,
#                      size=[1.2, 1.2, 0.8],
#                      color=[0.4, 0.8, 0.1],
#                      static=False, respondable=True)

for i in range(LOOPS):
  agent1 = NewQuadricopter(0,4)
  agent2 = NewQuadricopter(1,4)
  agent3 = NewQuadricopter(2,4)
  agent4 = NewQuadricopter(3,4)
  agent5 = NewQuadricopter(4,4)
  agent6 = NewQuadricopter(5,4)

  agents = [agent1,agent2,agent3,agent4,agent5,agent6]

  # Get a random position within a cuboid and set the target position
  #suction_cup.release()

  # pr.remove_model(m1)
  # [m,m1]= pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))

  #agent.drone_reset()
  #agent.set_3d_pose([0.0,0.0,1.5,0.0,0.0,0.0])
#ssssssss
  # agent1.set_3d_pose([random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max),1.7,0.0,0.0,0.0])
  # vx = agent1.get_drone_position()[0]
  # vy = agent1.get_drone_position()[1]
  
  # vpt1 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # while check_collision(vpt1,[vx,vy]):
  #   vpt1 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # agent2.set_3d_pose([vpt1[0],vpt1[1],1.7,0.0,0.0,0.0])

  # vpt2 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # while check_collision(vpt2,[vx,vy]) or check_collision(vpt2,vpt1) :
  #   vpt2 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # agent3.set_3d_pose([vpt2[0],vpt2[1],1.7,0.0,0.0,0.0])

  # vpt3 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # while check_collision(vpt3,[vx,vy]) or check_collision(vpt3,vpt1) or check_collision(vpt3,vpt2):
  #   vpt3 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # agent4.set_3d_pose([vpt3[0],vpt3[1],1.7,0.0,0.0,0.0])

  # vpt4 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # while check_collision(vpt4,[vx,vy]) or check_collision(vpt4,vpt1) or check_collision(vpt4,vpt2) or\
  # check_collision(vpt4,vpt3):
  #   vpt4 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # agent5.set_3d_pose([vpt4[0],vpt4[1],1.7,0.0,0.0,0.0])

  # vpt5 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # while check_collision(vpt5,[vx,vy]) or check_collision(vpt5,vpt1) or check_collision(vpt5,vpt2) or\
  # check_collision(vpt5,vpt3) or check_collision(vpt5,vpt4):
  #   vpt5 = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
  # agent6.set_3d_pose([vpt5[0],vpt5[1],1.7,0.0,0.0,0.0])
  saved_agents = []
  vpts = []
  for i in range(6):
      if i == 0:
          agents[i].set_3d_pose([random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max),1.7,0.0,0.0,0.0])
          vx = agents[i].get_drone_position()[0]
          vy = agents[i].get_drone_position()[1]
          vpts.append([vx,vy])
          saved_agents.append(i)
      else:
          #print(vpts,i)
          vpt = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
          check_list = [check_collision(vpt,vpts[m]) for m in saved_agents] #check_collision(vpt,vpts[m])
          check_conditions = np.sum(check_list)
          while check_conditions:
              vpt = [random.uniform(x_limit_min,x_limit_max),random.uniform(y_limit_min,y_limit_max)]
              check_list = [check_collision(vpt,vpts[m]) for m in saved_agents]
              check_conditions = np.sum(check_list)
              
          agents[i].set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
          vpts.append(vpt)
          saved_agents.append(i)
      
  #target.set_position([0.0,0.0,0.250])
  #target.set_orientation([0.0, 0.0, 0.0])
  
  img = (cam.capture_rgb() * 255).astype(np.uint8)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  video.write(img_rgb)

  
  vx1 = agent2.get_drone_position()[0]
  vy1 = agent2.get_drone_position()[1]
  vx2 = agent3.get_drone_position()[0]
  vy2 = agent3.get_drone_position()[1]
  vx3 = agent4.get_drone_position()[0]
  vy3 = agent4.get_drone_position()[1]
  vx4 = agent5.get_drone_position()[0]
  vy4 = agent5.get_drone_position()[1]
  vx5 = agent6.get_drone_position()[0]
  vy5 = agent6.get_drone_position()[1]

  pr.step()
  print(agent1.get_right_proximity())
  print(agent1.get_concentration())

  for j in np.arange(1000):
      
      vels = agent1.position_controller([vx,vy,1.7])
      vels1 = agent2.position_controller([vx1,vy1,1.7])
      vels2 = agent3.position_controller([vx2,vy2,1.7])
      vels3 = agent4.position_controller([vx3,vy3,1.7])
      vels4 = agent5.position_controller([vx4,vy4,1.7])
      vels5 = agent6.position_controller([vx5,vy5,1.7])
      

        #print(vels[0], vels[1], vels[2], vels[3])
      agent1.set_propller_velocity([vels[0], vels[1], vels[2], vels[3]])
      agent2.set_propller_velocity([vels1[0], vels1[1], vels1[2], vels1[3]])
      agent3.set_propller_velocity([vels2[0], vels2[1], vels2[2], vels2[3]])
      agent4.set_propller_velocity([vels3[0], vels3[1], vels3[2], vels3[3]])
      agent5.set_propller_velocity([vels4[0], vels4[1], vels4[2], vels4[3]])
      agent6.set_propller_velocity([vels5[0], vels5[1], vels5[2], vels5[3]])

        # Power each propeller
      agent1.control_propeller_thrust(1)
      agent1.control_propeller_thrust(2)
      agent1.control_propeller_thrust(3)
      agent1.control_propeller_thrust(4)

      agent2.control_propeller_thrust(1)
      agent2.control_propeller_thrust(2)
      agent2.control_propeller_thrust(3)
      agent2.control_propeller_thrust(4)

      agent3.control_propeller_thrust(1)
      agent3.control_propeller_thrust(2)
      agent3.control_propeller_thrust(3)
      agent3.control_propeller_thrust(4)

      agent4.control_propeller_thrust(1)
      agent4.control_propeller_thrust(2)
      agent4.control_propeller_thrust(3)
      agent4.control_propeller_thrust(4)

      agent5.control_propeller_thrust(1)
      agent5.control_propeller_thrust(2)
      agent5.control_propeller_thrust(3)
      agent5.control_propeller_thrust(4)

      agent6.control_propeller_thrust(1)
      agent6.control_propeller_thrust(2)
      agent6.control_propeller_thrust(3)
      agent6.control_propeller_thrust(4)
      
      #det = suction_cup.grasp(target)
          
      img = (cam.capture_rgb() * 255).astype(np.uint8)
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      video.write(img_rgb)
    
      pr.step()
      
      #print(np.r_[(target.get_position()[:]-np.array([0.0,0.0,1.5])),target.get_orientation()[:],agent.get_drone_position()[:], agent.velocities])    
    
video.release()

pr.stop()
pr.shutdown()



#robot_DIR = path.join(path.dirname(path.abspath(__file__)), 'models')
#m = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
#m1 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
#m2 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
#m3 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))

'''cam = VisionSensor('my_cam')
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

