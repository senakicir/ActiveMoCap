import cv2 as cv2
from math import radians, cos, sin, pi, degrees, acos, sqrt
import numpy as np
from helpers import range_angle, model_settings, shape_cov, MIDDLE_POSE_INDEX, FUTURE_POSE_INDEX, TOP_SPEED
import time as time 
from project_bones import take_potential_projection
import pdb 
from random import randint

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
SAFE_RADIUS = 14
TRAVEL = 0.1
TRAVEL2 = 4
UPPER_LIM = -6
LOWER_LIM = -1.5

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

    def get_delta_orient(self, target_yaw):
        delta_yaw = find_delta_yaw((self.drone_orientation)[2],  target_yaw)
        return delta_yaw

    def get_goal_pos_and_yaw_chosen(self, goal_state):
        goal_pos = goal_state["position"]
        #goal_yaw = goal_state["orientation"]
        #desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2],  goal_yaw)

        goal_vel = 5*(goal_pos-self.drone_pos)/np.linalg.norm((goal_pos-self.drone_pos))
        new_pos = self.drone_pos + goal_vel*DELTA_T
        _, new_phi = find_current_polar_info(new_pos, self.human_pos)
        new_orientation = new_phi+pi
        desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2],  new_orientation)
        return goal_vel , desired_yaw_deg       

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
    def __init__(self, pose_client, current_drone_pos):
        _, joint_names, _, _ = model_settings(pose_client.model)
        self.hip_index = joint_names.index('spine1')

        self.current_drone_pos = np.squeeze(pose_client.current_drone_pos)
        self.future_human_pos = pose_client.future_pose
        self.current_human_pos = pose_client.current_pose
        self.potential_states = []
        self.potential_hessians_normal = []
        self.potential_covs_normal = []
        self.current_state_ind = 0
        self.goal_state_ind =0
        self.potential_pose2d_list = []
        self.hessian_method = pose_client.hessian_method

    def get_potential_positions_really_spherical(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = TRAVEL

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.current_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec) #starts from 0?
        up_vec_norm = up_vec*travel/np.linalg.norm(up_vec)
        side_vec = np.cross(unit_z, new_drone_vec) 
        side_vec_norm = side_vec*travel/np.linalg.norm(side_vec)

        weights = [-1,0,1]
        ind = 0
        for w1 in weights:
            for w2 in weights:
                do_not_append = False
                if (w1==0 and w2 ==0):
                    norm_pos = new_drone_vec + self.current_human_pos[:, self.hip_index]
                    self.current_state_ind = ind
                else:
                    if (w1*w2 == 0):
                        pos = new_drone_vec + self.current_human_pos[:, self.hip_index] + up_vec_norm*w1 + side_vec_norm*w2
                    else:
                        pos = new_drone_vec + self.current_human_pos[:, self.hip_index] +  (up_vec_norm*w1 + side_vec_norm*w2)/sqrt(2)
                    potential_drone_vec = pos-self.current_human_pos[:, self.hip_index]
                    norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
                    norm_pos = norm_potential_drone_vec + self.current_human_pos[:, self.hip_index]
                
                if norm_pos[2]  > LOWER_LIM:
                    do_not_append = True
                if norm_pos[2]  < UPPER_LIM:
                    do_not_append = True
                
                if not do_not_append:
                    new_theta = acos((norm_pos[2] - self.current_human_pos[2, self.hip_index])/new_radius)
                    new_pitch = pi/2 -new_theta
                    _, new_phi = find_current_polar_info(norm_pos, self.current_human_pos[:, self.hip_index])
                    self.potential_states.append({"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch})
                ind += 1
        
        return self.potential_states

    def get_potential_positions_really_spherical_future(self):

        new_radius = SAFE_RADIUS
        UPPER_LIM = -7
        LOWER_LIM = -1
        unit_z = np.array([0,0,-1])
        travel = TRAVEL

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec) #starts from 0?
        side_vec = np.cross(unit_z, new_drone_vec) 
        up_vec_norm = up_vec*travel/np.linalg.norm(up_vec)
        side_vec_norm = side_vec*travel/np.linalg.norm(side_vec)

        weights_up = [-1,0,1]
        weights_side = [-1,0,1]
        if current_drone_pos[2]  > LOWER_LIM: #about to crash
            weights_up = [-1]
        elif current_drone_pos[2] + 1 > LOWER_LIM: #about to crash
            weights_up = [-1, 0]

        if current_drone_pos[2]  < UPPER_LIM:
            weights_up = [1]
        elif current_drone_pos[2] -1 < UPPER_LIM:
            weights_up = [0, 1]
                      
        ind = 0
        for w1 in weights_up:
            for w2 in weights_side:
                if (w1==0 and w2 ==0):
                    self.current_state_ind = ind
                    norm_pos = new_drone_vec + self.future_human_pos[:, self.hip_index]
                else:
                    if (w1*w2 == 0):
                        pos = new_drone_vec + self.future_human_pos[:, self.hip_index] + up_vec_norm*w1 + side_vec_norm*w2
                    else:
                        pos = new_drone_vec + self.future_human_pos[:, self.hip_index] +  (up_vec_norm*w1 + side_vec_norm*w2)/sqrt(2)
                    potential_drone_vec = pos-self.future_human_pos[:, self.hip_index]
                    norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
                    norm_pos = norm_potential_drone_vec + self.future_human_pos[:, self.hip_index]
                
                new_theta = acos((norm_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
                new_pitch = pi/2 -new_theta
                _, new_phi = find_current_polar_info(norm_pos, self.future_human_pos[:, self.hip_index])
                print("new z", norm_pos[2])
                self.potential_states.append({"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch})
                ind += 1
        return self.potential_states

    
    def constant_rotation_baseline(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = TRAVEL

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.current_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        side_vec = np.cross(unit_z, new_drone_vec) 
        side_vec_norm = side_vec*travel/np.linalg.norm(side_vec)
       
        pos = drone_vec + self.current_human_pos[:, self.hip_index] + side_vec_norm
        potential_drone_vec = pos-self.current_human_pos[:, self.hip_index]
        norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
        norm_pos = norm_potential_drone_vec + self.current_human_pos[:, self.hip_index]
   
        norm_pos[2] = self.current_human_pos[2, self.hip_index]-z_pos

        new_theta = acos((norm_pos[2] - self.current_human_pos[2, self.hip_index])/new_radius)
        new_pitch = pi/2 -new_theta
        _, new_phi = find_current_polar_info(norm_pos, self.current_human_pos[:, self.hip_index])
        goal_state = {"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch}

        return goal_state

    def constant_rotation_baseline_future(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = TRAVEL

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        side_vec = np.cross(unit_z, new_drone_vec) 
        side_vec_norm = side_vec*travel/np.linalg.norm(side_vec)
       
        pos = new_drone_vec + self.future_human_pos[:, self.hip_index] + side_vec_norm
        potential_drone_vec = pos - self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
        norm_pos = norm_potential_drone_vec + self.future_human_pos[:, self.hip_index]
   
        norm_pos[2] = self.current_human_pos[2, self.hip_index]-z_pos

        #new_theta = acos((norm_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        #new_pitch = pi/2 -new_theta
        #_, new_phi = find_current_polar_info(norm_pos, self.future_human_pos[:, self.hip_index])
        #goal_state = {"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch}

        goal_vel = TOP_SPEED*(norm_pos-self.current_drone_pos)/np.linalg.norm((norm_pos-self.current_drone_pos))
        new_pos = self.current_drone_pos + goal_vel*DELTA_T
        _, new_phi = find_current_polar_info(new_pos, self.future_human_pos[:, self.hip_index])
        new_orientation = new_phi+pi

        return goal_vel, new_orientation
    
    def find_hessians_for_potential_states(self, objective, pose_client, P_world):
        for potential_state_ind, potential_state in enumerate(self.potential_states):
            objective.reset(pose_client, potential_state)

            start_find_hess2 = time.time()
            hess2 = objective.hessian(P_world)
            end_find_hess2 = time.time()
            
            print("Time for finding hessian no", potential_state_ind, ": ", end_find_hess2-start_find_hess2)

            self.potential_hessians_normal.append(hess2)
            inv_hess2 = np.linalg.inv(hess2)

            if (self.hessian_method == 0):
                self.potential_covs_normal.append(shape_cov(inv_hess2, pose_client.model, FUTURE_POSE_INDEX))
            elif (self.hessian_method == 1):
                self.potential_covs_normal.append(shape_cov(inv_hess2, pose_client.model, MIDDLE_POSE_INDEX))
            else:
                self.potential_covs_normal.append(inv_hess2)

            #take projection 
            self.potential_pose2d_list.append(take_potential_projection(potential_state, pose_client.future_pose)) #sloppy
        return self.potential_covs_normal, self.potential_hessians_normal

    def find_best_potential_state(self):
        uncertainty_list = []
        for cov in self.potential_covs_normal:
            if self.hessian_method == 2:
                _, s, rotation = np.linalg.svd(cov)
                uncertainty_list.append(np.sum(s)) 
            else:
                uncertainty_list.append(np.linalg.det(cov))

        best_ind = uncertainty_list.index(min(uncertainty_list))
        self.goal_state_ind = best_ind
        print("uncertainty list:", uncertainty_list, "best ind", best_ind)
        goal_state = self.potential_states[best_ind]
        return goal_state

    def find_goal_vel_and_yaw(self, goal_state):
        goal_pos = goal_state["position"]
        goal_yaw = goal_state["orientation"]
        goal_vel = TOP_SPEED*(goal_pos-self.current_drone_pos)/np.linalg.norm((goal_pos-self.current_drone_pos))
        new_pos = self.current_drone_pos + goal_vel*DELTA_T
        _, new_phi = find_current_polar_info(new_pos, self.future_human_pos[:, self.hip_index])
        new_orientation = new_phi+pi
        return goal_vel, new_orientation


    def find_random_next_state(self):
        random_ind = randint(0, len(self.potential_states)-1)
        self.goal_state_ind = random_ind
        print("random ind", random_ind)
        return self.potential_states[random_ind]