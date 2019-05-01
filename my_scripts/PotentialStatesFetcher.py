from helpers import choose_frame_from_cov, FUTURE_POSE_INDEX, MIDDLE_POSE_INDEX, plot_potential_ellipses, plot_potential_projections, plot_potential_hessians, plot_potential_projections_noimage, euler_to_rotation_matrix, shape_cov_general, plot_potential_errors
import numpy as np
from State import find_current_polar_info, find_delta_yaw, SAFE_RADIUS
from determine_positions import objective_calib, objective_online
from math import radians, cos, sin, pi, degrees, acos, sqrt, inf
from random import randint
from project_bones import Projection_Client, C_cam_torch, CAMERA_ROLL_OFFSET, CAMERA_YAW_OFFSET, neat_tensor
import time as time
import torch

def sample_states_spherical(psf, new_radius, new_theta, new_phi):
    new_yaw = new_phi  + psf.human_orientation_GT
    x = new_radius*cos(new_yaw)*sin(new_theta) + psf.human_GT[0, psf.hip_index]
    y = new_radius*sin(new_yaw)*sin(new_theta) + psf.human_GT[1, psf.hip_index]
    z = new_radius*cos(new_theta)+ psf.human_GT[2, psf.hip_index]
    drone_pos = np.array([x, y, z])

    _, new_phi_go = find_current_polar_info(drone_pos, psf.human_GT[:, psf.hip_index]) #used to be norm_pos_go

    goal_state = PotentialState(position=drone_pos.copy(), orientation=new_phi_go+pi, pitch=new_theta+pi/2)

    psf.potential_states_try.append(goal_state)
    psf.potential_states_go.append(goal_state)

class PotentialState(object):
    def __init__(self, position, orientation, pitch):
        self.position = position
        self.orientation = orientation
        self.pitch = pitch

        self.potential_C_drone = torch.from_numpy(self.position[:, np.newaxis]).float()
        self.potential_R_drone = euler_to_rotation_matrix(0,0,self.orientation, returnTensor=True)
        self.potential_R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, self.pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor=True)

        drone_transformation = torch.cat((torch.cat((self.potential_R_drone, self.potential_C_drone), dim=1), neat_tensor), dim=0)
        camera_transformation = torch.cat((torch.cat((self.potential_R_cam, C_cam_torch), dim=1), neat_tensor), dim=0) 
        self.inv_transformation_matrix = torch.inverse(drone_transformation@camera_transformation)

    def get_goal_pos_yaw_pitch(self, curr_drone_orientation):
        desired_yaw_deg = find_delta_yaw((curr_drone_orientation)[2],  self.orientation)
        return self.position , desired_yaw_deg, self.pitch   

class PotentialStatesFetcher(object):
    def __init__(self, pose_client, active_parameters):
        _, self.joint_names, self.number_of_joints, self.hip_index = pose_client.model_settings()
        self.minmax = active_parameters["MINMAX"]
        self.hessian_part = active_parameters["HESSIAN_PART"]
        self.uncertainty_calc_method = active_parameters["UNCERTAINTY_CALC_METHOD"]
        self.wobble_freq = active_parameters["WOBBLE_FREQ"]
        self.updown_lim = active_parameters["UPDOWN_LIM"]
        self.target_z_pos = active_parameters["Z_POS"]
        self.lookahead = active_parameters["LOOKAHEAD"]
        self.go_distance = active_parameters["GO_DISTANCE"]

        self.trajectory = active_parameters["TRAJECTORY"]
        self.is_quiet = pose_client.quiet
        self.model = pose_client.model
        self.goUp = True
        self.POSITION_GRID = active_parameters["POSITION_GRID"]

        self.UPPER_LIM = active_parameters["UPPER_LIM"]
        self.LOWER_LIM = active_parameters["LOWER_LIM"]

        self.number_of_samples = len(self.POSITION_GRID)

        self.overall_error_list = np.zeros(self.number_of_samples)
        self.future_error_list = np.zeros(self.number_of_samples)
        self.error_std_list =  np.zeros(self.number_of_samples)
        self.uncertainty_list_whole = []
        self.uncertainty_list_future = []
        self.counter = 0 

    def reset(self, pose_client, current_state):
        self.current_drone_pos = np.squeeze(current_state.drone_pos_gt)
        self.human_GT = current_state.bone_pos_gt
        self.human_orientation_GT = current_state.human_orientation_gt

        self.future_human_pos = pose_client.future_pose
        self.current_human_pos = pose_client.current_pose
        self.potential_states_try = []
        self.potential_states_go = []
        self.potential_hessians_normal = []

        self.potential_covs_future = []
        self.potential_covs_middle = []
        self.potential_covs_whole = []

        self.optimized_poses = pose_client.optimized_poses
        self.optimized_traj = pose_client.optimized_traj

        self.current_state_ind = 0
        self.goal_state_ind =0

        self.potential_pose2d_list = []

        if (pose_client.isCalibratingEnergy):
            self.objective = objective_calib
        else:
            self.objective = objective_online

        self.overall_error_list = np.zeros(self.number_of_samples)
        self.future_error_list = np.zeros(self.number_of_samples)
        self.future_error_std_list =  np.zeros(self.number_of_samples)
        self.overall_error_std_list =  np.zeros(self.number_of_samples)
        self.uncertainty_list_whole = []
        self.uncertainty_list_future = []


    def get_potential_positions_really_spherical_future(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        side_vec = np.cross(unit_z, new_drone_vec) 
        up_vec_norm = up_vec*self.lookahead/np.linalg.norm(up_vec)
        side_vec_norm = side_vec*self.lookahead/np.linalg.norm(side_vec)
        up_vec_norm_go = up_vec* self.go_distance/np.linalg.norm(up_vec)
        side_vec_norm_go = side_vec* self.go_distance/np.linalg.norm(side_vec)

        weights_up = [-1,0,1]
        weights_side = [-1,0,1]
        if current_drone_pos[2]  > self.LOWER_LIM: #about to crash
            weights_up = [-1]
        elif current_drone_pos[2] + 1 > self.LOWER_LIM: #about to crash
            weights_up = [-1, 0]

        if current_drone_pos[2]  < self.UPPER_LIM:
            weights_up = [1]
        elif current_drone_pos[2] -1 < self.UPPER_LIM:
            weights_up = [0, 1]
                      
        ind = 0
        for w1 in weights_up:
            for w2 in weights_side:
                if (w1==0 and w2 ==0):
                    self.current_state_ind = ind
                    norm_pos = new_drone_vec + self.future_human_pos[:, self.hip_index]
                    norm_pos_go = norm_pos.copy()
                else:
                    pos = new_drone_vec + self.future_human_pos[:, self.hip_index] +  (up_vec_norm*w1 + side_vec_norm*w2)/sqrt(w1*w1+w2*w2)
                    pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + (up_vec_norm_go*w1 + side_vec_norm_go*w2)/sqrt(w1*w1+w2*w2)
                    potential_drone_vec = pos-self.future_human_pos[:, self.hip_index]
                    norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
                    norm_pos = norm_potential_drone_vec + self.future_human_pos[:, self.hip_index]

                    potential_drone_vec_go = pos_go-self.future_human_pos[:, self.hip_index]
                    norm_potential_drone_vec_go = potential_drone_vec_go * new_radius /np.linalg.norm(potential_drone_vec_go)
                    if w1 == 0:
                        norm_potential_drone_vec_go[2] = potential_drone_vec_go[2]
                    norm_pos_go = norm_potential_drone_vec_go + self.future_human_pos[:, self.hip_index]
                
                #if (w1 != 0 and w2 != 0):
                new_theta = acos((norm_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
                new_pitch = pi/2 -new_theta
                _, new_phi = find_current_polar_info(norm_pos, self.future_human_pos[:, self.hip_index])
                self.potential_states_try.append(PotentialState(position=norm_pos.copy(), orientation=new_phi+pi, pitch=new_pitch))

                new_theta_go = acos((norm_pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
                new_pitch_go = pi/2 -new_theta_go
                _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
                self.potential_states_go.append(PotentialState(position=norm_pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go))
                ind += 1

    def go_there(self, dir = "u"):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        side_vec = np.cross(unit_z, new_drone_vec) 
        up_vec_norm_go = up_vec*self.go_distance/np.linalg.norm(up_vec)
        side_vec_norm_go = side_vec*self.go_distance/np.linalg.norm(side_vec)

        if dir == "u":
            w1 = -1
            w2 = 0
                      
        pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + (up_vec_norm_go*w1 + side_vec_norm_go*w2)/sqrt(w1*w1+w2*w2)
        potential_drone_vec_go = pos_go-self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec_go = potential_drone_vec_go * new_radius /np.linalg.norm(potential_drone_vec_go)
        norm_pos_go = norm_potential_drone_vec_go + self.future_human_pos[:, self.hip_index]
    
        new_theta_go = acos((norm_pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch_go = pi/2 -new_theta_go
        _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = PotentialState(position=norm_pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go)
        return goal_state    

    def constant_rotation_baseline_future(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        side_vec = np.cross(unit_z, new_drone_vec) 
        side_vec_norm = side_vec* self.go_distance/np.linalg.norm(side_vec)
       
        pos = new_drone_vec + self.future_human_pos[:, self.hip_index] + side_vec_norm
        potential_drone_vec = pos - self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
        norm_pos = norm_potential_drone_vec + self.future_human_pos[:, self.hip_index]
   
        norm_pos[2] = self.target_z_pos

        new_theta = acos((norm_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch = pi/2 -new_theta
        _, new_phi = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = PotentialState(position=norm_pos.copy(), orientation=new_phi+pi, pitch=new_pitch)

        return goal_state

    def precalibration(self):
        new_radius = SAFE_RADIUS
        baseline_lim_up = self.updown_lim[0]
        baseline_lim_down = self.updown_lim[1]
        current_drone_pos = np.copy(self.current_drone_pos)

        if current_drone_pos[2] + 1 > baseline_lim_down: #about to crash
            self.goUp = True
        if current_drone_pos[2] -1 < baseline_lim_up:
            self.goUp = False

        if self.goUp:
            pos_go = current_drone_pos + np.array([0,0,-1])
        else:
            pos_go = current_drone_pos + np.array([0,0,1])

        new_theta_go = acos((pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch_go = pi/2 -new_theta_go
        _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = PotentialState(position=pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go)
        return goal_state

    def dome_experiment(self):
        for theta, phi in self.POSITION_GRID:
            sample_states_spherical(self, SAFE_RADIUS, theta, phi)
        return self.potential_states_try

    def up_down_baseline(self):
        new_radius = SAFE_RADIUS
        baseline_lim_up = self.updown_lim[0]
        baseline_lim_down = self.updown_lim[1]
        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        up_vec_norm_go = up_vec* self.go_distance/np.linalg.norm(up_vec)

        if current_drone_pos[2] + 1 > baseline_lim_down: #about to crash
            self.goUp = True
        if current_drone_pos[2] -1 < baseline_lim_up:
            self.goUp = False

        if self.goUp:
            pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + -up_vec_norm_go
        else:
            pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + up_vec_norm_go
        potential_drone_vec_go = pos_go-self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec_go = potential_drone_vec_go * new_radius /np.linalg.norm(potential_drone_vec_go)
        norm_pos_go = norm_potential_drone_vec_go + self.future_human_pos[:, self.hip_index]

        new_theta_go = acos((norm_pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch_go = pi/2 -new_theta_go
        _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index]) #used to be norm_pos_go
        goal_state = PotentialState(position=norm_pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go)
        return goal_state

    def left_right_baseline(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        wobble_lim_up = -6
        wobble_lim_down = -2

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        side_vec = np.cross(unit_z, new_drone_vec) 
        side_vec_norm_go = side_vec* self.go_distance/np.linalg.norm(side_vec)

        if current_drone_pos[2] + 1 > wobble_lim_down: #about to crash
            self.goUp = True
        if current_drone_pos[2] - 1 < wobble_lim_up:
            self.goUp = False

        if self.goUp:
            pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + (-up_vec_norm_go*up_vec_weight - side_vec_norm_go)/sqrt(up_vec_weight*up_vec_weight+1)
        else:
            pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + (up_vec_norm_go*up_vec_weight - side_vec_norm_go)/sqrt(up_vec_weight*up_vec_weight+1)
        potential_drone_vec_go = pos_go-self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec_go = potential_drone_vec_go * new_radius /np.linalg.norm(potential_drone_vec_go)
        norm_pos_go = norm_potential_drone_vec_go + self.future_human_pos[:, self.hip_index]

        new_theta_go = acos((norm_pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch_go = pi/2 -new_theta_go
        _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = PotentialState(position=norm_pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go)
        return goal_state

    def wobbly_baseline(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        wobble_lim_up = self.updown_lim[0]
        wobble_lim_down =self.updown_lim[1]
        up_vec_weight = self.wobble_freq

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        side_vec = np.cross(unit_z, new_drone_vec) 
        up_vec_norm_go = up_vec* self.go_distance/np.linalg.norm(up_vec)
        side_vec_norm_go = side_vec* self.go_distance/np.linalg.norm(side_vec)

        if current_drone_pos[2] + 1 > wobble_lim_down: #about to crash
            self.goUp = True
        if current_drone_pos[2] - 1 < wobble_lim_up:
            self.goUp = False

        if self.goUp:
            pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + (-up_vec_norm_go*up_vec_weight - side_vec_norm_go)/sqrt(up_vec_weight*up_vec_weight+1)
        else:
            pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] + (up_vec_norm_go*up_vec_weight - side_vec_norm_go)/sqrt(up_vec_weight*up_vec_weight+1)

        potential_drone_vec_go = pos_go-self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec_go = potential_drone_vec_go * new_radius /np.linalg.norm(potential_drone_vec_go)
        norm_pos_go = norm_potential_drone_vec_go + self.future_human_pos[:, self.hip_index]

        new_theta_go = acos((norm_pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch_go = pi/2 -new_theta_go
        _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = PotentialState(position=norm_pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go)

        return goal_state

    def constant_angle_baseline_future(self):

        new_radius = SAFE_RADIUS

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)
       
        pos = new_drone_vec + self.future_human_pos[:, self.hip_index] 
        potential_drone_vec = pos - self.future_human_pos[:, self.hip_index]
        norm_potential_drone_vec = potential_drone_vec * new_radius /np.linalg.norm(potential_drone_vec)
        norm_pos = norm_potential_drone_vec + self.future_human_pos[:, self.hip_index]
   
        norm_pos[2] = -1.8#self.current_human_pos[2, self.hip_index]-z_pos

        new_theta = acos((norm_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch = pi/2 -new_theta
        _, new_phi = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = PotentialState(position=norm_pos.copy(), orientation=new_phi+pi, pitch=new_pitch)

        return goal_state
    
    def find_hessians_for_potential_states(self, pose_client):
        self.counter += 1
        for potential_state_ind, potential_state in enumerate(self.potential_states_try):
            self.objective.reset_future(pose_client, potential_state)
            if pose_client.USE_TRAJECTORY_BASIS:
                hess2 = self.objective.hessian(self.optimized_traj)
            else:
                hess2 = self.objective.hessian(self.optimized_poses)
            self.potential_hessians_normal.append(hess2)

            inv_hess2 = np.linalg.inv(hess2)


            #if self.counter ==5 :
                #import IPython
                #IPython.embed()

                #hess_modified = hess2.copy()
                #for i in range(100,140):
                #    print(hess2[i,i])
                #    hess_modified[i,i] = 2
                #hess_modified[235,10] = 2
                #hess_modified[100:140,100:140] = 2
                #inv_hess_modified = np.linalg.inv(hess_modified)

                #import matplotlib.pyplot as plt
                #fig = plt.figure()
                #plt.subplot(131)
                #im1 = plt.imshow(inv_hess2)
                #plt.subplot(132)
                #fig.colorbar(im1)
                #im2 = plt.imshow(inv_hess_modified)
                #plt.subplot(133)
                #fig.colorbar(im2)
                #im3 = plt.imshow(np.sqrt(np.abs(inv_hess2-inv_hess_modified)))
                #fig.colorbar(im3)
                #plt.show()
                #plt.close()
 
            self.potential_covs_future.append(choose_frame_from_cov(inv_hess2, FUTURE_POSE_INDEX, self.number_of_joints))
            self.potential_covs_middle.append(choose_frame_from_cov(inv_hess2, MIDDLE_POSE_INDEX, self.number_of_joints))
            self.potential_covs_whole.append(inv_hess2)

            future_pose = torch.from_numpy(self.future_human_pos).float() 
            self.potential_pose2d_list.append(pose_client.projection_client.take_single_projection(future_pose, potential_state.inv_transformation_matrix))
        return self.potential_covs_whole, self.potential_hessians_normal

    def find_best_potential_state(self):
        for hessian_part_ind, covariances in enumerate([self.potential_covs_future, self.potential_covs_whole]):
            uncertainty_list = []
            for cov in covariances:
                if self.uncertainty_calc_method == "sum_eig":
                    _, s, _ = np.linalg.svd(cov)
                    uncertainty_list.append(np.sum(s)) 
                elif self.uncertainty_calc_method == "add_diag":
                    s = np.diag(cov)
                    uncertainty_list.append(np.sum(s)) 
                elif self.uncertainty_calc_method == "multip_eig":
                    _, s, _ = np.linalg.svd(cov)
                    uncertainty_list.append(s[0]*s[1]*s[2]) 
                    #import matplotlib.pyplot as plt
                    #fig =  plt.figure()
                    #print("largest three eigenvals", s[0], s[1], s[2])
                    #plt.plot(s, marker="^")
                    #plt.show()
                    #plt.close()
                elif self.uncertainty_calc_method == "determinant":
                    uncertainty_list.append(np.linalg.det(s))
                    #cov_shaped = shape_cov_general(cov, self.model, 0)
                    #uncertainty_joints = np.zeros([self.number_of_joints,])
                    #for joint_ind in range(self.number_of_joints):
                    #    _, s, _ = np.linalg.svd(cov_shaped[joint_ind, :, :])
                    #    uncertainty_joints[joint_ind] = np.sum(s)#np.linalg.det(cov_shaped[joint_ind, :, :]) 
                    #uncertainty_list.append(np.mean(uncertainty_joints))
                #elif self.uncertainty_calc_method == 4:
                  #  pass
            #if self.uncertainty_calc_method == "random":
             #   (uncertainty_lists[hessian_part_ind]) = (np.random.permutation(np.arange(self.number_of_samples))).tolist()
            if hessian_part_ind == 0:
                self.uncertainty_list_future = uncertainty_list.copy()
                final_list = uncertainty_list.copy()
            elif self.hessian_part == "whole":
                self.uncertainty_list_whole = uncertainty_list.copy()
                final_list = uncertainty_list.copy()

        if (self.minmax):
            best_ind = final_list.index(min(uncertainty_list))
        else:
            best_ind = final_list.index(max(uncertainty_list))
        self.goal_state_ind = best_ind
        print("uncertainty list var:", np.std(final_list), "uncertainty list min max", np.min(final_list), np.max(final_list), "best ind", best_ind)
        goal_state = self.potential_states_go[best_ind]
        return goal_state, best_ind

    def find_random_next_state(self):
        random_ind = randint(0, len(self.potential_states_go)-1)
        self.goal_state_ind = random_ind
        print("random ind", random_ind)
        return self.potential_states_go[random_ind]

    def plot_everything(self, linecount, plot_loc, photo_loc):
        if not self.is_quiet:
            #plot_potential_hessians(self.potential_covs_normal, linecount, plot_loc, custom_name = "potential_covs_normal_")
            #plot_potential_hessians(self.potential_hessians_normal, linecount, plot_loc, custom_name = "potential_hess_normal_")
            #plot_potential_states(pose_client.current_pose, pose_client.future_pose, bone_pos_3d_GT, potential_states, C_drone, R_drone, pose_client.hip_index, plot_loc, airsim_client.linecount)
            #plot_potential_projections(self.potential_pose2d_list, linecount, plot_loc, photo_loc, self.bone_connections)
            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=False, top_down=False, plot_errors=True)
            plot_potential_ellipses(self, plot_loc, linecount, ellipses=True, top_down=True, plot_errors=False)
            plot_potential_errors(self, plot_loc, linecount)

    def plot_projections(self, linecount, plot_loc):
        plot_potential_projections_noimage(self.potential_pose2d_list, linecount, plot_loc, self.joint_names)
