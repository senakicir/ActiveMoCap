import cv2 as cv2
from math import radians, cos, sin, pi, degrees, acos
import numpy as np
from helpers import range_angle, model_settings, shape_cov
import time as time 
from project_bones import take_potential_projection

#constants
BETA = 0.35
DRONE_POS_IND = 0
HUMAN_POS_IND = 2
R_SHOULDER_IND = 3
L_SHOULDER_IND = 4
DRONE_ORIENTATION_IND = 1

INCREMENT_DEGREE_AMOUNT = radians(-20)
INCREMENT_RADIUS = 3

z_pos = -0.5
DELTA_T = 0.2
N = 4.0
TIME_HORIZON = N*DELTA_T
SAFE_RADIUS = 14


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

    def get_desired_pos_and_angle_fixed_rotation(self):
        desired_polar_angle = self.current_degree + INCREMENT_DEGREE_AMOUNT
        desired_polar_pos = np.array([cos(desired_polar_angle) * self.radius, sin(desired_polar_angle) * self.radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel 
        desired_pos[2] = self.human_pos[2]-z_pos
        desired_yaw = self.current_degree + INCREMENT_DEGREE_AMOUNT/N + pi
        desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2], desired_yaw)

        return desired_pos, desired_yaw_deg

    def get_goal_pos_and_yaw_active(self, pose_client):
        goal_state = pose_client.goal_state
        goal_pos = goal_state["position"]
        goal_yaw = goal_state["orientation"]
        desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2],  goal_yaw)
        return goal_pos , desired_yaw_deg

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
        
class Potential_States_Fetcher(object):
    def __init__(self, current_drone_pos, current_human_pos, future_human_pos, model):
        _, joint_names, _, _ = model_settings(model)
        self.hip_index = joint_names.index('spine1')

        self.current_drone_pos = np.squeeze(current_drone_pos)
        self.future_human_pos = future_human_pos
        self.current_human_pos = current_human_pos
        self.potential_states = []
        self.potential_hessians_normal = []
        self.potential_covs_normal = []
        self.current_state_ind = 0
        self.goal_state_ind =0
        self.potential_pose2d_list = []

    def get_potential_positions_shifting_rad(self):
        neutral_drone_pos = np.copy(self.current_drone_pos + (self.future_human_pos[:, self.hip_index] - self.current_human_pos[:, self.hip_index]))
        _, neutral_yaw = find_current_polar_info(neutral_drone_pos, self.future_human_pos[:, self.hip_index])
       
        projected_distance_vect = neutral_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(projected_distance_vect[0:2,])
        print("current radius", cur_radius)

        delta_radius_list = [-INCREMENT_RADIUS, 0, INCREMENT_RADIUS]

        if cur_radius < 13:
            delta_radius_list = [INCREMENT_RADIUS]
        elif cur_radius -INCREMENT_RADIUS < 13:
            delta_radius_list = [0, INCREMENT_RADIUS]

        if cur_radius > 20:
            delta_radius_list = [-INCREMENT_RADIUS]
        elif cur_radius + INCREMENT_RADIUS:
            delta_radius_list = [-INCREMENT_RADIUS, 0]

        delta_angle_list = [radians(-20), 0, radians(20)]

        for delta_ang in delta_angle_list:
            for delta_rad in delta_radius_list:
                new_yaw = neutral_yaw + delta_ang 
                new_rad = cur_radius + delta_rad
                x = new_rad*cos(new_yaw) + self.future_human_pos[0, self.hip_index]
                y = new_rad*sin(new_yaw) + self.future_human_pos[1, self.hip_index]
                drone_pos = np.array([x, y, neutral_drone_pos[2]])
                self.potential_states.append({"position":np.copy(drone_pos), "orientation": new_yaw+pi})
        return self.potential_states

    def get_potential_positions_shifting_z(self):
        neutral_drone_pos = np.copy(self.current_drone_pos + (self.future_human_pos[:, self.hip_index] - self.current_human_pos[:, self.hip_index]))
        _, neutral_yaw = find_current_polar_info(neutral_drone_pos, self.future_human_pos[:, self.hip_index])
       
        new_rad = SAFE_RADIUS
        current_z = neutral_drone_pos[2]
        INCREMENT_Z = -2 #makes them rise

        delta_z_list = [-INCREMENT_Z, 0, INCREMENT_Z] #fall stay rise
        self.current_state = 4
        print("current_z", current_z)

        if current_z > -1: #ABOUT TO CRASH
            delta_z_list = [INCREMENT_Z]
        elif current_z - INCREMENT_Z > -1:
            delta_z_list = [0, INCREMENT_Z]

        if current_z < -5:
            delta_z_list = [-INCREMENT_Z]
        elif current_z + INCREMENT_Z < -5:
            delta_z_list = [-INCREMENT_Z, 0]

        print("delta_z_list", delta_z_list, )
        delta_angle_list = [radians(-20), 0, radians(20)]

        for delta_z in delta_z_list:
            for delta_ang in delta_angle_list:
                new_yaw = neutral_yaw + delta_ang 
                new_z = current_z + delta_z
                x = new_rad*cos(new_yaw) + self.future_human_pos[0, self.hip_index]
                y = new_rad*sin(new_yaw) + self.future_human_pos[1, self.hip_index]
                drone_pos = np.array([x, y, new_z])
                self.potential_states.append({"position":np.copy(drone_pos), "orientation": new_yaw+pi})
        return self.potential_states

    def get_potential_positions_spherical(self):
        neutral_drone_pos = np.copy(self.current_drone_pos + (self.future_human_pos[:, self.hip_index] - self.current_human_pos[:, self.hip_index]))
        _, neutral_yaw = find_current_polar_info(neutral_drone_pos, self.future_human_pos[:, self.hip_index])
        projected_distance_vect = neutral_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(projected_distance_vect)
        
        new_rad = SAFE_RADIUS
        UPPER_LIM = -5
        LOWER_LIM = 0.5
        
        current_z = neutral_drone_pos[2]
        print("current_z: ", current_z, "current_radius: ", cur_radius)

        current_theta = acos((current_z- self.future_human_pos[2, self.hip_index])/cur_radius)

        new_theta_list = [current_theta+radians(-20), current_theta, current_theta+radians(20)]
        #new_phi_list = [neutral_yaw+radians(-30), neutral_yaw+radians(-20), neutral_yaw, neutral_yaw+radians(20), neutral_yaw+radians(30)]
        new_phi_list = [neutral_yaw+radians(-20), neutral_yaw, neutral_yaw+radians(20)]
        self.current_state_ind = 4

        if current_z > LOWER_LIM: #ABOUT TO CRASH
            new_theta_list = [current_theta+radians(20)]
            self.current_state_ind = 1
        elif current_z + 1 > LOWER_LIM:
            new_theta_list = [current_theta, current_theta+radians(20)]
            self.current_state_ind = 1
            
        if current_z < UPPER_LIM:
            new_theta_list = [current_theta+radians(-20)]
            self.current_state_ind = 1
        elif current_z - 1  < UPPER_LIM:
            new_theta_list = [current_theta+radians(-20), current_theta]
            self.current_state_ind = 4

        for new_theta in new_theta_list:
            for new_phi in new_phi_list:
                x = new_rad*cos(new_phi)*sin(new_theta) + self.future_human_pos[0, self.hip_index]
                y = new_rad*sin(new_phi)*sin(new_theta) + self.future_human_pos[1, self.hip_index]
                z = new_rad*cos(new_theta)+ self.future_human_pos[2, self.hip_index]
                drone_pos = np.array([x, y, z])
                new_pitch = pi/2 -new_theta
                self.potential_states.append({"position":np.copy(drone_pos), "orientation": new_phi+pi, "pitch": new_pitch})
                difference = np.linalg.norm(self.current_drone_pos - drone_pos)
                print("new z: ", z, "difference: ", difference)
        return self.potential_states

    def get_potential_positions_cartesian(self):
        x_list = [-2, 0, 2]
        y_list = [-2, 0, 2]
        neutral_drone_pos = np.copy(self.current_drone_pos + (self.future_human_pos[:, self.hip_index] - self.current_human_pos[:, self.hip_index]))
        for x in x_list:
            for y in y_list:
                drone_pos = np.array([neutral_drone_pos[0] + x, neutral_drone_pos[1] + y, neutral_drone_pos[2]])
                _, polar_degree = find_current_polar_info(drone_pos, self.future_human_pos[:, self.hip_index])
                self.potential_states.append({"position":np.copy(drone_pos), "orientation": polar_degree+pi})
        return self.potential_states

    def find_hessians_for_potential_states(self, objective, pose_client, P_world):
        for potential_state_ind, potential_state in enumerate(self.potential_states):
            objective.reset(pose_client, potential_state)

            #start_find_hess1 = time.time()
            #hess1 = objective.mini_hessian(P_world)
            #end_find_hess1 = time.time()
            start_find_hess2 = time.time()
            hess2 = objective.hessian(P_world)
            end_find_hess2 = time.time()
            #start_find_hess3 = time.time()
            #hess3 = objective.mini_hessian_hip(P_world)
            #end_find_hess3 = time.time()
            #print("Time for finding hessian no", potential_state_ind, ": ", end_find_hess1-start_find_hess1, end_find_hess2-start_find_hess2, end_find_hess3-start_find_hess3)
            print("Time for finding hessian no", potential_state_ind, ": ", end_find_hess2-start_find_hess2)

            # hess2_reshaped = shape_cov_test2(hess2, pose_client.model)

            #potential_hessians_mini.append(hess1)
            self.potential_hessians_normal.append(hess2)
            #potential_hessians_hip.append(hess3)

            #inv_hess1 = np.linalg.inv(hess1)
            inv_hess2 = np.linalg.inv(hess2)
            #inv_hess3 = np.linalg.inv(hess3)

            #potential_covs_mini.append(shape_cov_mini(inv_hess1, pose_client.model, 0))
            self.potential_covs_normal.append(shape_cov(inv_hess2, pose_client.model, 0))
            #self.potential_covs_normal.append(inv_hess2)
            #potential_covs_hip.append(shape_cov_hip(inv_hess3, pose_client.model, 0))

            #take projection 
            self.potential_pose2d_list.append(take_potential_projection(potential_state, pose_client.future_pose)) #sloppy
        return self.potential_covs_normal, self.potential_hessians_normal


    def find_best_potential_state(self):
        uncertainty_list = []
        for cov in self.potential_covs_normal:
            uncertainty_list.append(np.trace(cov))

        best_ind = uncertainty_list.index(min(uncertainty_list))
        self.goal_state_ind = best_ind
        print("best ind", best_ind)
        return self.potential_states[best_ind]
