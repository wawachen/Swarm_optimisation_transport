from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.object import Object
from pyrep.backend import sim
from pyrep.objects.shape import Shape
import numpy as np
from typing import List, Union
from pyrep.const import ObjectType
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.sensors.accelerometer import Accelerometer
from pyrep.sensors.gyroscope import Gyroscope
from pyrep.sensors.spherical_vision_sensor import SphericalVisionSensor
from pyrep.drone_controller_utility.velocity_control_loop import VelocityController
from pyrep.drone_controller_utility.acceleration_control_loop import AccelerationController
from pyrep.drone_controller_utility.attitude_control_loop import AttitudeController
from pyrep.drone_controller_utility.position_control_loop import PositionController
from pyrep.objects.dummy import Dummy

class NewDrone_base(Object):
    """ This new class is based on the quadricopter model in Coppeliasim."""

    def __init__(self, count:int, num_propeller:int, name:str):

        force_sensor_names = ['%s_propeller%s' % (name, str(i + 1)) for i in range(num_propeller)]
        respondable_names = ['%s_propeller_respondable%s' % (name, str(i+1)) for i in range(num_propeller)]
        suffix = '' if count == 0 else '#%d' % (count - 1)

        #get the handle of the drone base
        super().__init__(name + suffix)

        self._num_propeller = num_propeller

        #get handles of force sensor
        self.force_sensors = [ForceSensor(fname + suffix).get_handle() for fname in force_sensor_names]
        #get the handles of propeller respondable
        self.respondables = [Shape(sname+suffix)._handle for sname in respondable_names]
        self.panoramic_holder = Shape('sphericalVisionRGBAndDepth'+suffix)
        self.proximity_holder = Dummy('ProximityDummy'+suffix)

        #get handles of vision sensor
        self.vs_floor = VisionSensor('Vision_sensor_down'+suffix) 
        self.panoramic_camera = SphericalVisionSensor('sphericalVisionRGBAndDepth',suffix) 

        #get handles of ultrosonic sensor
        self.depth_sensor = ProximitySensor('Ultrasonic_sensor'+suffix)

        self.front_sensor = ProximitySensor('Proximity_sensor_front'+suffix)
        self.back_sensor = ProximitySensor('Proximity_sensor_back'+suffix)
        self.left_sensor = ProximitySensor('Proximity_sensor_left'+suffix)
        self.right_sensor = ProximitySensor('Proximity_sensor_right'+suffix)

        #get handles of IMU
        self.accSensor = Accelerometer('Accelerometer',suffix)
        self.gyroSensor = Gyroscope('GyroSensor',suffix)

        #add some simulation parameters
        s = sim.simGetObjectSizeFactor(self.force_sensors[0]) # current size factor 

        self.particleCountPerSecond = 430
        self.particleDensity = 8500
        self.particleSize = 1 * 0.005 * s
        self.notFullParticles = [0,0,0,0]
        self.pre_v = [0,0,0,0]  # previous size factor
        self.velocities = np.array([6.1,6.1,6.1,6.1])

        #------for test suction cup
        # self.particlesTargetVelocities = [0,0,0,0]
        # self.pParam = 2
        # self.iParam = 0
        # self.dParam = 0
        # self.vParam = -2

        # self.cumul = 0
        # self.lastE = 0
        # self.pAlphaE = 0
        # self.pBetaE = 0
        # self.psp2 = 0
        # self.psp1 = 0
        
        # self.prevEuler = 0 

        #-------------------
        self.drone_position_controller = PositionController()
        self.drone_velocity_controller = VelocityController()
        self.drone_acceleration_controller = AccelerationController()
        self.drone_attitude_controller = AttitudeController()

    def get_3d_pose(self) -> np.ndarray:
        """Gets the ground truth 3D pose of the robot [x, y, z, yaw, pitch, roll].
         
        :return: A List containing the x, y, z, roll, pitch, yaw (in radians).
        """
        return np.r_[self.get_position()[:], self.get_orientation()[:]]

    def get_velocities(self):
        """ get the linear and angular velocity """
        return self.get_velocity()
    

    def set_3d_pose(self, pose: Union[List[float], np.ndarray]) -> None:
        """Sets the 3D pose of the robot [x, y, z, yaw, pitch, roll]

        :param pose: A List containing the x, y, z, roll, pitch, yaw (in radians).
        """
        x, y, z, roll, pitch, yaw = pose
        self.set_position([x, y, z])
        self.set_orientation([roll, pitch, yaw])

    def get_picture(self):

        rgb_fl = self.vs_floor.capture_rgb()  #return: A numpy array of size (width, height, 3)
        depth_fl = self.vs_floor.capture_depth() #return: A numpy array of size (width, height)

        return  [rgb_fl, depth_fl]
    
    def get_panoramic(self):
        rgb_fl = self.panoramic_camera.capture_rgb()  #return: A numpy array of size (width, height, 3)
        return rgb_fl

    def set_drone_position(self, pos):
        self.set_position(pos)

    def set_drone_orientation(self, ori):
        self.set_orientation(ori)

    def get_drone_position(self):
        return self.get_position()[:]

    def get_drone_orientation(self):
        return self.get_orientation()[:]
    
    def get_concentration(self):
        return self.depth_sensor.read()[0]

    def get_detection(self,obj):
        return self.depth_sensor.is_detected(obj)

    def get_IMUdata(self):
        """ get the data from accelerometer and gyroscope """
        return [self.accSensor.read(), self.gyroSensor.read()]
    
    def get_right_proximity(self):
        """It returns the data of proximity sensor in front, back, left, right"""
        return self.right_sensor.read()

    def get_left_proximity(self):
        """It returns the data of proximity sensor in front, back, left, right"""
        return self.left_sensor.read()

    def get_back_proximity(self):
        """It returns the data of proximity sensor in front, back, left, right"""
        return self.back_sensor.read()

    def get_front_proximity(self):
        """It returns the data of proximity sensor in front, back, left, right"""
        return self.front_sensor.read()
        
    def control_propeller_thrust(self, num_p) -> None:
        """ set thrust for the particular propeller, num_p is the number of the propeller(1 to 4) """

        particleVelocity = self.velocities[num_p-1]
        
        ts = sim.simGetSimulationTimeStep() 
    
        m = sim.simGetObjectMatrix(self.force_sensors[num_p-1], -1)

        requiredParticleCnt = self.particleCountPerSecond * ts + self.notFullParticles[num_p-1]
        self.notFullParticles[num_p-1] = requiredParticleCnt % 1
        requiredParticleCnt = np.floor(requiredParticleCnt)

        totalExertedForce = requiredParticleCnt * self.particleDensity * particleVelocity * np.pi * self.particleSize * self.particleSize * self.particleSize/(6*ts)
        force = [0,0,totalExertedForce]
        m[3] = 0
        m[7] = 0
        m[11] = 0
        force = sim.simMultiplyVector(m,force)
        
        torque = [0, 0, pow(-1,num_p)*0.002 * particleVelocity]
        torque = sim.simMultiplyVector(m,torque)

        sim.simAddForceAndTorque(self.respondables[num_p-1], force, torque) 
        
    def set_propller_velocity(self,v) -> None:
        """ set the motor velocities for the propellers """
        self.velocities = v
       
    def _get_requested_type(self) -> ObjectType:
        """Gets the type of the object.

        :return: Type of the object.
        """
        return ObjectType(sim.simGetObjectType(self.get_handle()))

    def drone_reset(self):
        self.notFullParticles = [0,0,0,0]
        self.velocities = np.array([6.1,6.1,6.1,6.1])
        self.particlesTargetVelocities = [0,0,0,0]
        self.cumul = 0

    def position_controller(self,pos):
        wp_cmd = [pos[0],pos[1],pos[2],0.0]
        vel_cmd = self.drone_position_controller.CalculatePositionControl(wp_cmd,self.get_drone_orientation(),self.get_drone_position())
        acc_cmd = self.drone_velocity_controller.CalculateVelocityControl(vel_cmd, self.get_drone_orientation(), self.get_velocities()[0])
        att_cmd = self.drone_acceleration_controller.CalculateAccelerationControl(acc_cmd[:3],acc_cmd[3],self.get_drone_orientation(),self.get_IMUdata()[0]) 

        anrate_cmd = self.drone_attitude_controller.CalculateAttitudeControl(att_cmd, self.get_drone_orientation())
        ctr_cmd = self.drone_attitude_controller.CalculateRateControl(anrate_cmd, self.get_drone_orientation(), self.get_IMUdata()[1])
        des_motorvel = self.drone_attitude_controller.CalculateMotorCommands(ctr_cmd)

        particlesTargetVelocities = np.zeros(4)
        particlesTargetVelocities[0] = des_motorvel[0]
        particlesTargetVelocities[1] = des_motorvel[1]
        particlesTargetVelocities[2] = des_motorvel[2]
        particlesTargetVelocities[3] = des_motorvel[3]
    
        return particlesTargetVelocities
    
    def velocity_controller(self,vx,vy):
        wp_cmd_p = self.get_drone_position()[:2]
        wp_cmd = [wp_cmd_p[0],wp_cmd_p[1],1.7,0.0] #change to 1.7 for RL test
        vel_cmd = self.drone_position_controller.CalculatePositionControl(wp_cmd,self.get_drone_orientation(),self.get_drone_position())

        vel_cmd[0] = vx
        vel_cmd[1] = vy
        acc_cmd = self.drone_velocity_controller.CalculateVelocityControl(vel_cmd, self.get_drone_orientation(), self.get_velocities()[0])
        att_cmd = self.drone_acceleration_controller.CalculateAccelerationControl(acc_cmd[:3],acc_cmd[3],self.get_drone_orientation(),self.get_IMUdata()[0]) 

        anrate_cmd = self.drone_attitude_controller.CalculateAttitudeControl(att_cmd, self.get_drone_orientation())
        ctr_cmd = self.drone_attitude_controller.CalculateRateControl(anrate_cmd, self.get_drone_orientation(), self.get_IMUdata()[1])
        des_motorvel = self.drone_attitude_controller.CalculateMotorCommands(ctr_cmd)

        particlesTargetVelocities = np.zeros(4)
        particlesTargetVelocities[0] = des_motorvel[0]
        particlesTargetVelocities[1] = des_motorvel[1]
        particlesTargetVelocities[2] = des_motorvel[2]
        particlesTargetVelocities[3] = des_motorvel[3]
    
        return particlesTargetVelocities

    def velocity_controller1(self,vx,vy,vz=None):
        wp_cmd = [0,0,1.7,0.0] #change to 1.7 for RL test
        vel_cmd = self.drone_position_controller.CalculatePositionControl(wp_cmd,self.get_drone_orientation(),self.get_drone_position())

        vel_cmd[0] = vx
        vel_cmd[1] = vy

        if vz is not None:
            vel_cmd[2] = vz

        acc_cmd = self.drone_velocity_controller.CalculateVelocityControl(vel_cmd, self.get_drone_orientation(), self.get_velocities()[0])
        att_cmd = self.drone_acceleration_controller.CalculateAccelerationControl(acc_cmd[:3],acc_cmd[3],self.get_drone_orientation(),self.get_IMUdata()[0]) 

        anrate_cmd = self.drone_attitude_controller.CalculateAttitudeControl(att_cmd, self.get_drone_orientation())
        ctr_cmd = self.drone_attitude_controller.CalculateRateControl(anrate_cmd, self.get_drone_orientation(), self.get_IMUdata()[1])
        des_motorvel = self.drone_attitude_controller.CalculateMotorCommands(ctr_cmd)

        particlesTargetVelocities = np.zeros(4)
        particlesTargetVelocities[0] = des_motorvel[0]
        particlesTargetVelocities[1] = des_motorvel[1]
        particlesTargetVelocities[2] = des_motorvel[2]
        particlesTargetVelocities[3] = des_motorvel[3]
    
        return particlesTargetVelocities
    
    def rpythrust_controller(self,action):
        att_cmd = np.zeros([4])
        att_cmd[0] = action[0]
        att_cmd[1] = action[1]
        att_cmd[2] = 0
        att_cmd[3] = action[2]

        anrate_cmd = self.drone_attitude_controller.CalculateAttitudeControl(att_cmd, self.get_drone_orientation())
        ctr_cmd = self.drone_attitude_controller.CalculateRateControl(anrate_cmd, self.get_drone_orientation(), self.get_IMUdata()[1])
        des_motorvel = self.drone_attitude_controller.CalculateMotorCommands(ctr_cmd)

        particlesTargetVelocities = np.zeros(4)
        particlesTargetVelocities[0] = des_motorvel[0]
        particlesTargetVelocities[1] = des_motorvel[1]
        particlesTargetVelocities[2] = des_motorvel[2]
        particlesTargetVelocities[3] = des_motorvel[3]
    
        return particlesTargetVelocities

    #------------------------------For test suction cup
    # def simple_controller(self, target_loc):
    #     '''simple PID controller'''

    #     sp = [0.0,0.0,0.0]
    #     euler = [0.0,0.0,0.0]
    #     l = []
    #     angu = []
    
    #     targetPos = target_loc
    #     #print("target position is: ",target_pos)
    #     pos = self.get_3d_pose()[:3]

    #     l, angu = sim.simGetVelocity(self._handle)
    #     e = (targetPos[2]-pos[2])
    #     self.cumul = self.cumul + e
    #     pv = self.pParam*e
    #     thrust = 5.335 + pv + self.iParam*self.cumul + self.dParam * (e-self.lastE) + l[2]*self.vParam
    #     self.lastE = e
    
    #     for i in range(3):
    #         sp[i] = targetPos[i]-pos[i]
        
    #     m= sim.simGetObjectMatrix(self._handle,-1)
    #     vx = [1,0,0]
    #     vx = sim.simMultiplyVector(m,vx)
    #     vy = [0,1,0]
    #     vy = sim.simMultiplyVector(m,vy)

    #     alphaE = (vy[2]-m[11])
    #     alphaCorr = 0.25*alphaE + 2.1*(alphaE-self.pAlphaE)
    #     betaE = (vx[2]-m[11])
    #     betaCorr = -0.25*betaE-2.1*(betaE-self.pBetaE)
    #     self.pAlphaE = alphaE
    #     self.pBetaE = betaE
    #     alphaCorr = alphaCorr + sp[1]*0.005 + 1*(sp[1]-self.psp2)
    #     betaCorr = betaCorr-sp[0]*0.005 - 1*(sp[0]-self.psp1)
    #     self.psp2 = sp[1]
    #     self.psp1 = sp[0]
    
    #     eulert1 = self.get_3d_pose()[3:]
    #     eulert2 = [0.0,0.0,0.0]

    #     for i in range(3):
    #         euler[i] = eulert1[i]-eulert2[i]
        
    #     rotCorr = euler[2]*0.1 + 2*(euler[2]-self.prevEuler)
    #     self.prevEuler = euler[2]
    
    #     self.particlesTargetVelocities[0] = thrust*(1-alphaCorr+betaCorr+rotCorr)
    #     self.particlesTargetVelocities[1] = thrust*(1-alphaCorr-betaCorr-rotCorr)
    #     self.particlesTargetVelocities[2] = thrust*(1+alphaCorr-betaCorr+rotCorr)
    #     self.particlesTargetVelocities[3] = thrust*(1+alphaCorr+betaCorr-rotCorr)
    
    #     return self.particlesTargetVelocities



    