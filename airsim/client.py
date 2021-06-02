from __future__ import print_function

from .utils import *
from .types import *
import airsim
import msgpackrpc #install as admin: pip install msgpack-rpc-python
import numpy as np #pip install numpy
import msgpack
import time
import math
import logging
import torch


class VehicleClient:
    def __init__(self, length_of_simulation, ip = "", port = 41451, timeout_value = 3600):
        if (ip == ""):
            ip = "127.0.0.1"
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')

        #sena was here
        self.program_started = False #sena was here
        self.DRONE_INITIAL_POS = np.array([0, 0, 0])
        self.default_initial_anim_time = 1
        self.requiredEstimationData = []
        self.isCalibratingEnergy = False
        self.linecount = 0 #sena was here
        self.online_linecount = 0
        self.length_of_simulation = length_of_simulation

        self.is_using_airsim = True
        self.end = False

        SIZE_X = 1024
        SIZE_Y = 576
        focal_length = SIZE_X/2
        px = SIZE_X/2
        py = SIZE_Y/2
        flip_x_y = torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]])
        K_torch =  torch.FloatTensor([[focal_length,0,px],[0,focal_length,py],[0,0,1]])
        self.intrinsics = {"f":focal_length,"size_x":SIZE_X, "size_y":SIZE_Y, "K_torch": K_torch, "flip_x_y": flip_x_y}
        
    # -----------------------------------  Common vehicle APIs ---------------------------------------------
    def reset(self):
         #sena was here
        self.DRONE_INITIAL_POS = np.array([0, 0, 0])
        self.program_started = False
        self.linecount = 0
        self.client.call('reset')
        time.sleep(1)
        for _ in range(1):
            self.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
            time.sleep(1)

    #sena was here
    def increment_linecount(self, is_calibrating_energy):
        self.linecount += 1
        if not is_calibrating_energy:
            self.online_linecount += 1

    def ping(self):
        return self.client.call('ping')

    def getClientVersion(self):
        return 1 # sync with C++ client
    def getServerVersion(self):
        return self.client.call('getServerVersion')
    def getMinRequiredServerVersion(self):
        return 1 # sync with C++ client
    def getMinRequiredClientVersion(self):
        return self.client.call('getMinRequiredClientVersion')

    # basic flight control
    def enableApiControl(self, is_enabled, vehicle_name = ''):
        return self.client.call('enableApiControl', is_enabled, vehicle_name)
    def isApiControlEnabled(self, vehicle_name = ''):
        return self.client.call('isApiControlEnabled', vehicle_name)
    def armDisarm(self, arm, vehicle_name = ''):
        return self.client.call('armDisarm', arm, vehicle_name)
 
    def simPause(self, is_paused, loop_mode):
        if loop_mode == "flight_simulation" or "try_controller_control":
            self.client.call('simPause', is_paused)

    def setAnimationTime(self, time):
        self.client.call('setAnimationTime', time)
    def getAnimationTime(self):
        return self.client.call('getAnimationTime')
    def changeAnimation(self, newAnimNum, vehicle_name = ''):
        self.client.call('changeAnimation', vehicle_name, newAnimNum)
        self.client.call('setAnimationTime', self.default_initial_anim_time)
        time.sleep(1)


    def simIsPause(self):
        return self.client.call("simIsPaused")

    def simContinueForTime(self, seconds):
        self.client.call('simContinueForTime', seconds)

    def getHomeGeoPoint(self, vehicle_name = ''):
        return GeoPoint.from_msgpack(self.client.call('getHomeGeoPoint', vehicle_name))

    def confirmConnection(self):
        if self.ping():
            print("Connected!")
        else:
             print("Ping returned false!")
        server_ver = self.getServerVersion()
        client_ver = self.getClientVersion()
        server_min_ver = self.getMinRequiredServerVersion()
        client_min_ver = self.getMinRequiredClientVersion()
    
        ver_info = "Client Ver:" + str(client_ver) + " (Min Req: " + str(client_min_ver) + \
              "), Server Ver:" + str(server_ver) + " (Min Req: " + str(server_min_ver) + ")"

        if server_ver < server_min_ver:
            print(ver_info, file=sys.stderr)
            print("AirSim server is of older version and not supported by this client. Please upgrade!")
        elif client_ver < client_min_ver:
            print(ver_info, file=sys.stderr)
            print("AirSim client is of older version and not supported by this server. Please upgrade!")
        else:
            print(ver_info)
        print('')

    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImage(self, camera_name, image_type, vehicle_name = ''):
        # todo: in future remove below, it's only for compatibility to pre v1.2
        camera_name = str(camera_name)

        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        result = self.client.call('simGetImage', camera_name, image_type, vehicle_name)
        if (result == "" or result == "\0"):
            return None
        return result

    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImages(self, requests, vehicle_name = ''):
        responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        return [ImageResponse.from_msgpack(response_raw) for response_raw in responses_raw]

    def simGetCollisionInfo(self, vehicle_name = ''):
        return CollisionInfo.from_msgpack(self.client.call('simGetCollisionInfo', vehicle_name))

    def simSetVehiclePose(self, pose, vehicle_name = ''):
        #sena was here
        position = Vector3r(x_val=pose.position[0], y_val=pose.position[1], z_val=pose.position[2])
        orientation = to_quaternion(roll=0, pitch=0, yaw=pose.orientation)
        go_pose = Pose(position_val=position, orientation_val = orientation)
        ignore_collison = True
        self.client.call('simSetVehiclePose', go_pose, ignore_collison, vehicle_name)

    def simGetVehiclePose(self, vehicle_name = ''):
        pose = self.client.call('simGetVehiclePose', vehicle_name)
        return Pose.from_msgpack(pose)
    def simGetObjectPose(self, object_name):
        pose = self.client.call('simGetObjectPose', object_name)
        return Pose.from_msgpack(pose)

    def simSetSegmentationObjectID(self, mesh_name, object_id, is_name_regex = False):
        return self.client.call('simSetSegmentationObjectID', mesh_name, object_id, is_name_regex)
    def simGetSegmentationObjectID(self, mesh_name):
        return self.client.call('simGetSegmentationObjectID', mesh_name)
    def simPrintLogMessage(self, message, message_param = "", severity = 0):
        return self.client.call('simPrintLogMessage', message, message_param, severity)

    def simGetCameraInfo(self, camera_name, vehicle_name = ''):
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        return CameraInfo.from_msgpack(self.client.call('simGetCameraInfo', str(camera_name), vehicle_name))
    def simSetCameraOrientation(self, camera_name, orientation, vehicle_name = ''):
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        self.client.call('simSetCameraOrientation', str(camera_name), orientation, vehicle_name)

    def simGetGroundTruthKinematics(self, vehicle_name = ''):
        return self.client.call('simGetGroundTruthKinematics', vehicle_name)
    simGetGroundTruthKinematics.__annotations__ = {'return': KinematicsState}
    def simGetGroundTruthEnvironment(self, vehicle_name = ''):
        return self.client.call('simGetGroundTruthEnvironment', vehicle_name)
    simGetGroundTruthEnvironment.__annotations__ = {'return': EnvironmentState}

    def cancelLastTask():
        self.client.call('cancelLastTask')

    def waitOnLastTask(timeout_sec = float('nan')):
        return self.client.call('waitOnLastTask', timeout_sec)


    #sena was here
    def initInitialDronePos(self, vehicle_name = ''):
        if (self.program_started == False):
            self.program_started = True
            initial_drone_pos = self.client.call('getInitialDronePos', vehicle_name)
            self.DRONE_INITIAL_POS =  np.array([initial_drone_pos['x_val'], initial_drone_pos['y_val'], -initial_drone_pos['z_val']])

    # legacy handling
    # TODO: remove below legacy wrappers in future major releases
    upgrade_api_help = "\nPlease see https://github.com/Microsoft/AirSim/blob/master/docs/upgrade_apis.md for more info."
    def simGetPose(self):
        logging.warning("simGetPose API is renamed to simGetVehiclePose. Please update your code." + self.upgrade_api_help)
        return self.simGetVehiclePose()
    def simSetPose(self, pose, ignore_collison):
        logging.warning("simSetPose API is renamed to simSetVehiclePose. Please update your code." + self.upgrade_api_help)
        return self.simSetVehiclePose(pose, ignore_collison)
    def getCollisionInfo(self):
        logging.warning("getCollisionInfo API is renamed to simGetCollisionInfo. Please update your code." + self.upgrade_api_help)
        return self.simGetCollisionInfo()
    def getCameraInfo(self, camera_id):
        logging.warning("getCameraInfo API is renamed to simGetCameraInfo. Please update your code." + self.upgrade_api_help)
        return self.simGetCameraInfo(camera_id)
    def setCameraOrientation(self, camera_id, orientation):
        logging.warning("setCameraOrientation API is renamed to simSetCameraOrientation. Please update your code." + self.upgrade_api_help)
        return self.simSetCameraOrientation(camera_id, orientation)
    def getPosition(self):
        logging.warning("getPosition API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthKinematics().position
    def getVelocity(self):
        logging.warning("getVelocity API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthKinematics().linear_velocity
    def getOrientation(self):
        logging.warning("getOrientation API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthKinematics().orientation
    def getLandedState(self):
        raise Exception("getLandedState API is deprecated. Please use getMultirotorState() API")
    def getGpsLocation(self):
        logging.warning("getGpsLocation API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthEnvironment().geo_point
    def takeoff(self, max_wait_seconds = 15):
        raise Exception("takeoff API is deprecated. Please use takeoffAsync() API." + self.upgrade_api_help)
    def land(self, max_wait_seconds = 60):
        raise Exception("land API is deprecated. Please use landAsync() API." + self.upgrade_api_help)
    def goHome(self):
        raise Exception("goHome API is deprecated. Please use goHomeAsync() API." + self.upgrade_api_help)
    def hover(self):
        raise Exception("hover API is deprecated. Please use hoverAsync() API." + self.upgrade_api_help)
    def moveByAngleZ(self, pitch, roll, z, yaw, duration):
        raise Exception("moveByAngleZ API is deprecated. Please use moveByAngleZAsync() API." + self.upgrade_api_help)
    def moveByAngleThrottle(self, pitch, roll, throttle, yaw_rate, duration):
        raise Exception("moveByAngleThrottle API is deprecated. Please use moveByAngleThrottleAsync() API." + self.upgrade_api_help)
    def moveByVelocity(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        raise Exception("moveByVelocity API is deprecated. Please use moveByVelocityAsync() API." + self.upgrade_api_help)
    def moveByVelocityZ(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        raise Exception("moveByVelocityZ API is deprecated. Please use moveByVelocityZAsync() API." + self.upgrade_api_help)
    def moveOnPath(self, path, velocity, max_wait_seconds = 60, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        raise Exception("moveOnPath API is deprecated. Please use moveOnPathAsync() API." + self.upgrade_api_help)
    def moveToZ(self, z, velocity, max_wait_seconds = 60, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        raise Exception("moveToZ API is deprecated. Please use moveToZAsync() API." + self.upgrade_api_help)
    def moveToPosition(self, x, y, z, velocity, max_wait_seconds = 60, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        raise Exception("moveToPosition API is deprecated. Please use moveToPositionAsync() API." + self.upgrade_api_help)
    def moveByManual(self, vx_max, vy_max, z_min, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        raise Exception("moveByManual API is deprecated. Please use moveByManualAsync() API." + self.upgrade_api_help)
    def rotateToYaw(self, yaw, max_wait_seconds = 60, margin = 5):
        raise Exception("rotateToYaw API is deprecated. Please use rotateToYawAsync() API." + self.upgrade_api_help)
    def rotateByYawRate(self, yaw_rate, duration):
        raise Exception("rotateByYawRate API is deprecated. Please use rotateByYawRateAsync() API." + self.upgrade_api_help)
    def setRCData(self, rcdata = RCData()):
        raise Exception("setRCData API is deprecated. Please use moveByRC() API." + self.upgrade_api_help)

    

# -----------------------------------  Multirotor APIs ---------------------------------------------
class MultirotorClient(VehicleClient, object):
    def __init__(self, length_of_simulation, port):
        #sena was here
        super(MultirotorClient, self).__init__(length_of_simulation, port=port)

    def takeoffAsync(self, timeout_sec = 20, vehicle_name = ''):
        return self.client.call_async('takeoff', timeout_sec, vehicle_name)  
    def landAsync(self, timeout_sec = 60, vehicle_name = ''):
        return self.client.call_async('land', timeout_sec, vehicle_name)   
    def goHomeAsync(self, timeout_sec = 3e+38, vehicle_name = ''):
        return self.client.call_async('goHome', timeout_sec, vehicle_name)

    # APIs for control
    def moveByAngleZAsync(self, pitch, roll, z, yaw, duration, vehicle_name = ''):
        return self.client.call_async('moveByAngleZ', pitch, roll, z, yaw, duration, vehicle_name)
    def moveByAngleThrottleAsync(self, pitch, roll, throttle, yaw_rate, duration, vehicle_name = ''):
        return self.client.call_async('moveByAngleThrottle', pitch, roll, throttle, yaw_rate, duration, vehicle_name)
    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        return self.client.call_async('moveByVelocity', vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name)
    def moveByVelocityZAsync(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        return self.client.call_async('moveByVelocityZ', vx, vy, z, duration, drivetrain, yaw_mode, vehicle_name)
    def moveOnPathAsync(self, path, velocity, timeout_sec = 3e+38, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), 
        lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        return self.client.call_async('moveOnPath', path, velocity, timeout_sec, drivetrain, yaw_mode, lookahead, adaptive_lookahead, vehicle_name)
    def moveToPositionAsync(self, x, y, z, velocity, timeout_sec = 3e+38, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), 
        lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        return self.client.call_async('moveToPosition', x, y, z, velocity, timeout_sec, drivetrain, yaw_mode, lookahead, adaptive_lookahead, vehicle_name)
    def moveToZAsync(self, z, velocity, timeout_sec = 3e+38, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        return self.client.call_async('moveToZ', z, velocity, timeout_sec, yaw_mode, lookahead, adaptive_lookahead, vehicle_name)
    def moveByManualAsync(self, vx_max, vy_max, z_min, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        """Read current RC state and use it to control the vehicles. 

        Parameters sets up the constraints on velocity and minimum altitude while flying. If RC state is detected to violate these constraints
        then that RC state would be ignored.

        :param vx_max: max velocity allowed in x direction
        :param vy_max: max velocity allowed in y direction
        :param vz_max: max velocity allowed in z direction
        :param z_min: min z allowed for vehicle position
        :param duration: after this duration vehicle would switch back to non-manual mode
        :param drivetrain: when ForwardOnly, vehicle rotates itself so that its front is always facing the direction of travel. If MaxDegreeOfFreedom then it doesn't do that (crab-like movement)
        :param yaw_mode: Specifies if vehicle should face at given angle (is_rate=False) or should be rotating around its axis at given rate (is_rate=True)
        """
        return self.client.call_async('moveByManual', vx_max, vy_max, z_min, duration, drivetrain, yaw_mode, vehicle_name)
    def rotateToYawAsync(self, yaw, timeout_sec = 3e+38, margin = 5, vehicle_name = ''):
        return self.client.call_async('rotateToYaw', yaw, timeout_sec, margin, vehicle_name)
    def rotateByYawRateAsync(self, yaw_rate, duration, vehicle_name = ''):
        return self.client.call_async('rotateByYawRate', yaw_rate, duration, vehicle_name)
    def hoverAsync(self, vehicle_name = ''):
        return self.client.call_async('hover', vehicle_name)

    def moveByRC(self, rcdata = RCData(), vehicle_name = ''):
        return self.client.call('moveByRC', rcdata, vehicle_name)
        
    # query vehicle state
    def getMultirotorState(self, vehicle_name = ''):
        return MultirotorState.from_msgpack(self.client.call('getMultirotorState', vehicle_name))
    getMultirotorState.__annotations__ = {'return': MultirotorState}


# -----------------------------------  Car APIs ---------------------------------------------
class CarClient(VehicleClient, object):
    def __init__(self):
        super(CarClient, self).__init__()

    def setCarControls(self, controls, vehicle_name = ''):
        self.client.call('setCarControls', controls, vehicle_name)

    def getCarState(self, vehicle_name = ''):
        state_raw = self.client.call('getCarState', vehicle_name)
        return CarState.from_msgpack(state_raw)
