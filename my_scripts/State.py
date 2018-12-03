import cv2 as cv2
from math import radians, cos, sin, pi, degrees, acos, sqrt
import numpy as np
from helpers import range_angle, model_settings, shape_cov, MIDDLE_POSE_INDEX, FUTURE_POSE_INDEX
import time as time 
from project_bones import take_potential_projection
import pdb 

#constants
BETA = 0.35
DRONE_POS_IND = 0
HUMAN_POS_IND = 2
R_SHOULDER_IND = 3
L_SHOULDER_IND = 4
DRONE_ORIENTATION_IND = 1

INCREMENT_DEGREE_AMOUNT = radians(-20)
INCREMENT_RADIUS = 3

z_pos = -1.5
DELTA_T = 0.2
N = 4.0
TIME_HORIZON = N*DELTA_T
SAFE_RADIUS = 7
TOP_SPEED = 5

def find_current_polar_info(drone_pos, human_pos):
    polar_pos = drone_pos - human_pos  #subtrack the human_pos in order to find the current polar position vector.
    polar_degree = np.arctan2(polar_pos[1], polar_pos[0])  #NOT relative to initial human angle, not using currently
    return polar_pos, polar_degree

def find_delta_yaw(current_yaw, desired_yaw):
    current_yaw_deg = degrees(current_yaw)
    yaw_candidates = np.array([degrees(desired_yaw), degrees(desired_yaw) - 360, degrees(desired_yaw) +360])
    min_diff = np.array([abs(current_yaw_deg -  yaw_candidates[0]), abs(current_yaw_deg -  yaw_candidates[1]), abs(current_yaw_deg -  yaw_candidates[2])])
    return yaw_candidates[np.argmin(min_diff)]


class State(object):
    def __init__(self):
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
        #self.human_rotation_speed = 0
        self.human_pos = np.array([0,0,0])
        self.human_vel = 0
        self.human_speed = 0
        self.drone_pos = np.array([0,0,0])
        self.current_polar_pos = np.array([0,0,0])
        self.current_degree = 0
        self.drone_orientation = np.array([0,0,0])
        self.radius = SAFE_RADIUS#np.linalg.norm(projected_distance_vect[0:2,]) #to do
        self.prev_human_pos = -42

        drone_polar_pos = np.array([0,0,0])#positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        #self.some_angle = range_angle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)

    def updateState(self, positions_):
        self.positions = positions_
        self.human_pos  = self.positions[HUMAN_POS_IND,:]
        if np.all(self.prev_human_pos == -42):
            self.human_vel = np.zeros([3,])
        else:
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


    def get_desired_pos_and_angle_fixed_rotation(self):
        desired_polar_angle = self.current_degree + INCREMENT_DEGREE_AMOUNT
        desired_polar_pos = np.array([cos(desired_polar_angle) * self.radius, sin(desired_polar_angle) * self.radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel 
        desired_pos[2] = self.human_pos[2]-z_pos
        desired_yaw = self.current_degree + pi#INCREMENT_DEGREE_AMOUNT/N + pi
        desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2], desired_yaw)

        return desired_pos, desired_yaw_deg

    def get_delta_orient(self, target_yaw):
        delta_yaw = find_delta_yaw((self.drone_orientation)[2],  target_yaw)
        return delta_yaw

    def get_goal_pos_yaw_pitch(self, goal_state):
        goal_pos = goal_state["position"]
        goal_yaw = goal_state["orientation"]
        cam_pitch = goal_state["pitch"]
        desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2],  goal_yaw)
        return goal_pos , desired_yaw_deg, cam_pitch   

    def get_current_pitch(self):
        new_radius = np.linalg.norm(self.drone_pos - self.human_pos)
        new_theta = acos((self.drone_pos[2] - self.human_pos[2])/new_radius)
        new_pitch = pi/2 - new_theta
        return new_pitch

    def get_desired_pos_and_yaw_trackbar(self):
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