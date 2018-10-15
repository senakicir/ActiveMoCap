import cv2 as cv2
from math import radians, cos, sin, pi, degrees
import numpy as np
from helpers import range_angle, model_settings

#constants
BETA = 0.35
DRONE_POS_IND = 0
HUMAN_POS_IND = 2
R_SHOULDER_IND = 3
L_SHOULDER_IND = 4
DRONE_ORIENTATION_IND = 1

INCREMENT_DEGREE_AMOUNT = radians(-20)

z_pos = 0.8
DELTA_T = 0.2
N = 4.0
TIME_HORIZON = N*DELTA_T

def find_current_polar_info(drone_pos, human_pos):
    polar_pos = drone_pos - human_pos  #subtrack the human_pos in order to find the current polar position vector.
    polar_degree = np.arctan2(polar_pos[1], polar_pos[0])  #NOT relative to initial human angle, not using currently

    return polar_pos, polar_degree

class State(object):
    def __init__(self, positions_):
        self.positions = positions_
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
        #self.human_rotation_speed = 0
        self.human_pos = positions_[HUMAN_POS_IND,:]
        self.human_vel = 0
        self.human_speed = 0
        self.drone_pos = np.array([0,0,0])
        self.current_polar_pos = np.array([0,0,0])
        self.current_degree = 0
        self.drone_orientation = np.array([0,0,0])
        projected_distance_vect = positions_[HUMAN_POS_IND, :]
        self.radius = np.linalg.norm(projected_distance_vect[0:2,]) #to do
        self.prev_human_pos = 0

        drone_polar_pos = - positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        self.some_angle = range_angle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)

    def updateState(self, positions_):
        self.positions = positions_
        self.human_pos  = self.positions[HUMAN_POS_IND,:]
        self.human_vel =  (self.human_pos - self.prev_human_pos)/DELTA_T
        self.prev_human_pos = self.human_pos
        self.human_speed = np.linalg.norm(self.human_vel) #the speed of the human (scalar)
        
        #what angle and polar position is the drone at currently
        self.drone_pos = positions_[DRONE_POS_IND, :] #airsim gives us the drone coordinates with initial drone loc. as origin
        self.drone_orientation = positions_[DRONE_ORIENTATION_IND, :]

        self.current_polar_pos , self.current_degree  = find_current_polar_info(self.drone_pos, self.human_pos)
 
        #calculate human orientation
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #prev_human_orientation = self.human_orientation
        #a filter to eliminate noisy data (smoother movement)
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])*BETA + prev_human_orientation*(1-BETA)
        #self.human_rotation_speed = (self.human_orientation-prev_human_orientation)/DELTA_T


    def getDesiredPosAndAngle(self):
        desired_polar_angle = self.current_degree + INCREMENT_DEGREE_AMOUNT
        print("desired polar angle", degrees(desired_polar_angle))
        desired_polar_pos = np.array([cos(desired_polar_angle) * self.radius, sin(desired_polar_angle) * self.radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,z_pos]) - np.array([0,0,1])        
        desired_yaw = self.current_degree + INCREMENT_DEGREE_AMOUNT/N + pi
        #desired_yaw = desired_polar_angle
        #print ("desired_polar_pos", desired_polar_pos)
        return desired_pos, desired_yaw

    def getDesiredPosAndAngleTrackbar(self):
        #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
        input_rad = radians(cv2.getTrackbarPos('Angle', 'Drone Control')) #according to what degree we want the drone to be at
        current_radius = cv2.getTrackbarPos('Radius', 'Drone Control')
        desired_z_pos = cv2.getTrackbarPos('Z', 'Drone Control')
        #input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
        #desired_polar_angle = state.human_orientation + input_rad + state.human_rotation_speed*TIME_HORIZON
        desired_polar_angle = input_rad

        desired_polar_pos = np.array([cos(desired_polar_angle) * current_radius, sin(desired_polar_angle) * current_radius, 0])
        #desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,desired_z_pos])
        desired_pos = desired_polar_pos + self.human_pos - np.array([0,0,desired_z_pos])
        desired_yaw = desired_polar_angle - pi
        return desired_pos, desired_yaw
        
class Potential_States_Fetcher(object):
    def __init__(self, current_drone_pos, current_human_pos, future_human_pos, model):
        _, joint_names, _, _ = model_settings(model)
        self.hip_index = joint_names.index('spine1')

        self.current_drone_pos = np.squeeze(current_drone_pos)
        self.future_human_pos = future_human_pos
        self.current_human_pos = current_human_pos
        self.potential_positions = []

    def get_potential_positions(self):
        delta_radius_list = [-2, 0, 2]
        delta_angle_list = [radians(-20), 0, radians(20)]
        neutral_drone_pos = np.copy(self.current_drone_pos + (self.future_human_pos[:, self.hip_index] - self.current_human_pos[:, self.hip_index]))
        _, neutral_yaw = find_current_polar_info(neutral_drone_pos, self.future_human_pos[:, self.hip_index])
       
        projected_distance_vect = neutral_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(projected_distance_vect[0:2,])

        for delta_ang in delta_angle_list:
            for delta_rad in delta_radius_list:
                new_yaw = neutral_yaw + delta_ang 
                new_rad = cur_radius + delta_rad
                x = new_rad*cos(new_yaw) + self.future_human_pos[0, self.hip_index]
                y = new_rad*sin(new_yaw) + self.future_human_pos[1, self.hip_index]
                drone_pos = np.array([x, y, neutral_drone_pos[2]])
                self.potential_positions.append({"position":np.copy(drone_pos), "orientation": new_yaw+pi})
        return self.potential_positions

    def get_potential_positions_cartesian(self):
        x_list = [-2, 0, 2]
        y_list = [-2, 0, 2]
        neutral_drone_pos = np.copy(self.current_drone_pos + (self.future_human_pos[:, self.hip_index] - self.current_human_pos[:, self.hip_index]))
        for x in x_list:
            for y in y_list:
                drone_pos = np.array([neutral_drone_pos[0] + x, neutral_drone_pos[1] + y, neutral_drone_pos[2]])
                _, polar_degree = find_current_polar_info(drone_pos, self.future_human_pos[:, self.hip_index])
                self.potential_positions.append({"position":np.copy(drone_pos), "orientation": polar_degree+pi})
        return self.potential_positions



