from helpers import range_angle, choose_frame_from_cov,plot_potential_ellipses, plot_potential_projections, plot_potential_hessians, plot_potential_projections_noimage, euler_to_rotation_matrix, shape_cov_general, plot_potential_errors, plot_potential_errors_and_uncertainties, plot_potential_trajectories, plot_dome, plot_potential_uncertainties
import numpy as np
from State import find_current_polar_info, find_delta_yaw
from determine_positions import objective_calib, objective_online
from math import radians, cos, sin, pi, degrees, acos, sqrt, inf
from project_bones import Projection_Client, CAMERA_ROLL_OFFSET, CAMERA_YAW_OFFSET, neat_tensor
from crop import is_pose_in_image
from motion_predictor import Motion_Predictor
import time as time
import torch


key_indices= {"c":0, "l":1, "r":2, "u":3, "d":4, "lu":5, "ld":6, "ru":7, "rd":8} 
key_weights= {"c":np.array([0,0]), "l":np.array([-1,0]), "r":np.array([1,0]), "u":np.array([0,-1]),
              "d":np.array([0,1]), "lu":np.array([-0.707,-0.707]), "ld":np.array([-0.707,0.707]), 
              "ru":np.array([0.707,-0.707]), "rd":np.array([0.707,0.707])} 


def sample_states_spherical(psf, new_radius, new_theta, new_phi, viewpoint_ind, C_cam_torch):
    new_yaw = new_phi  + 0#psf.human_orientation_GT
    x = new_radius*cos(new_yaw)*sin(new_theta) + psf.human_GT[0, psf.hip_index]
    y = new_radius*sin(new_yaw)*sin(new_theta) + psf.human_GT[1, psf.hip_index]
    z = new_radius*cos(new_theta)+ psf.human_GT[2, psf.hip_index]
    drone_pos = np.array([x, y, z])

    _, new_phi_go = find_current_polar_info(drone_pos, psf.human_GT[:, psf.hip_index]) #used to be norm_pos_go

    a_potential_state = PotentialState(position=drone_pos.copy(), orientation=new_phi_go+pi, pitch=new_theta+pi/2, index=viewpoint_ind,  C_cam_torch=C_cam_torch)
    return a_potential_state

class PotentialState(object):
    def __init__(self, position, orientation, pitch, index, C_cam_torch):
        self.position = position
        self.orientation = orientation
        self.pitch = pitch
        self.index = index
        self.camera_id = 0
        self.C_cam_torch = C_cam_torch

        self.potential_C_drone = torch.from_numpy(self.position[:, np.newaxis]).float()
        self.potential_R_drone = euler_to_rotation_matrix(0,0,self.orientation, returnTensor=True)
        self.potential_R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, self.pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor=True)
        self.potential_projection = None

        drone_transformation = torch.cat((torch.cat((self.potential_R_drone, self.potential_C_drone), dim=1), neat_tensor), dim=0)
        camera_transformation = torch.cat((torch.cat((self.potential_R_cam, self.C_cam_torch), dim=1), neat_tensor), dim=0) 
        self.inv_transformation_matrix = torch.inverse(drone_transformation@camera_transformation)

    def get_goal_yaw_pitch(self, curr_drone_orientation):
        desired_yaw_deg = find_delta_yaw((curr_drone_orientation)[2],  self.orientation)
        return desired_yaw_deg*1, self.pitch   
    
    def deep_copy_state(self):
        new_state = PotentialState(self.position, self.orientation, self.pitch, self.index, self.C_cam_torch)
        new_state.potential_projection = self.potential_projection.clone()
        return new_state

    def get_potential_projection(self, future_human_pose, projection_client):
        self.potential_projection = projection_client.take_single_projection(future_human_pose, self.inv_transformation_matrix, self.camera_id)
        pose_in_image = is_pose_in_image(self.potential_projection, projection_client.size_x, projection_client.size_y)
        #pose_in_image = True
        return pose_in_image

    def print_state(self):
        print("State index:"+str(index)+". Position:", self.potential_C_drone)

class PotentialState_External_Dataset(object):
    def __init__(self, transformation_matrix, index, camera_id):
        self.transformation_matrix = transformation_matrix
        self.position = self.transformation_matrix[0:3,3]
        self.inv_transformation_matrix = torch.inverse(self.transformation_matrix)
        self.index = index
        self.pitch = 0
        self.camera_id=camera_id
        self.potential_projection = None

        self.potential_C_drone = self.position.unsqueeze(1)

    def get_goal_yaw_pitch(self, curr_drone_orientation):
        return  0, 0 

    def get_potential_projection(self, future_human_pose, projection_client):
        self.potential_projection = projection_client.take_single_projection(future_human_pose, self.inv_transformation_matrix, self.camera_id)
        pose_in_image = is_pose_in_image(self.potential_projection, projection_client.size_x, projection_client.size_y)
        return pose_in_image

    def deep_copy_state(self):
        new_state = PotentialState_External_Dataset(self.transformation_matrix, self.index, self.camera_id)
        new_state.potential_projection = self.potential_projection.clone()
        return new_state

class Potential_Trajectory(object):
    def __init__(self, index, future_window_size, direction):
        self.future_window_size = future_window_size
        self.trajectory_index = index
        self.states = {}
        self.inv_transformation_matrix = torch.zeros(self.future_window_size, 4, 4)
        self.drone_positions = torch.zeros(self.future_window_size, 3, 1)
        #self.pitches = torch.zeros(self.future_window_size,)
        self.potential_hessian = np.zeros([1,1])
        self.potential_cov_dict = {"whole":np.zeros([1,1]), "future":np.zeros([1,1]), "middle":np.zeros([1,1])}
        self.uncertainty = 42
        self.errors_middle_dict = {}
        self.errors_overall_dict = {}
        for future_ind in range(future_window_size):
            self.errors_middle_dict[future_ind] = []
            self.errors_overall_dict[future_ind] = []    
        self.error_middle = 42
        self.error_overall = 42
        self.cam_list = []    
        self.potential_2d_poses = torch.zeros([self.future_window_size, 2, 15])
        self.direction = direction

    def get_movement_direction(self, future_index):
        return self.direction[future_index, :]

    def append_to_traj(self, future_ind, potential_state):
        self.states[self.future_window_size-future_ind-1] = potential_state.deep_copy_state()
        self.inv_transformation_matrix[self.future_window_size-future_ind-1, :, :] = potential_state.inv_transformation_matrix
        self.drone_positions[self.future_window_size-future_ind-1, :, :] = potential_state.potential_C_drone
        self.cam_list.append(potential_state.camera_id)
        assert potential_state.potential_projection is not None
        self.potential_2d_poses[self.future_window_size-future_ind-1, :, :] = potential_state.potential_projection

    def set_cov(self, potential_hessian, future_pose_index, middle_pose_index, number_of_joints):
        self.potential_hessian = potential_hessian.copy()
        inv_hessian = np.linalg.inv(self.potential_hessian)

        self.potential_cov_dict["future"] = choose_frame_from_cov(inv_hessian, future_pose_index, number_of_joints)
        self.potential_cov_dict["middle"] = choose_frame_from_cov(inv_hessian, middle_pose_index, number_of_joints)
        self.potential_cov_dict["whole"] = inv_hessian
    
    def find_uncertainty(self, method, part):
        cov = self.potential_cov_dict[part]

        if method == "sum_eig":
            _, s, _ = np.linalg.svd(cov)
            self.uncertainty = (np.sum(s)) 
        elif method == "add_diag":
            s = np.diag(cov)
            self.uncertainty = (np.sum(s))
        elif method == "multip_eig":
            _, s, _ = np.linalg.svd(cov)
            self.uncertainty = s[0]*s[1]*s[2]
            #import matplotlib.pyplot as plt
            #fig =  plt.figure()
            #print("largest three eigenvals", s[0], s[1], s[2])
            #plt.plot(s, marker="^")
            #plt.show()
            #plt.close()
        elif method == "determinant":
            self.uncertainty = np.linalg.det(s)
            #cov_shaped = shape_cov_general(cov, self.model, 0)
            #uncertainty_joints = np.zeros([self.number_of_joints,])
            #for joint_ind in range(self.number_of_joints):
            #    _, s, _ = np.linalg.svd(cov_shaped[joint_ind, :, :])
            #    uncertainty_joints[joint_ind] = np.sum(s)#np.linalg.det(cov_shaped[joint_ind, :, :]) 
            #uncertainty_list.append(np.mean(uncertainty_joints))
        elif method == "max_eig":
            _, s, _ = np.linalg.svd(cov)
            self.uncertainty = (np.max(s)) 
        elif method == "root_six":
            _, s, _ = np.linalg.svd(cov)
            self.uncertainty = np.sum(np.power(s, 1/6)) 

    def record_error_for_trial(self, future_ind, middle_error, overall_error):
        self.errors_middle_dict[future_ind].append(middle_error)
        #self.errors_overall_dict[future_ind].append(overall_error[0])

    def find_overall_error(self):
        error_middle = 0
        #error_overall = 0
        for future_ind in range(self.future_window_size):
            error_middle += sum(self.errors_middle_dict[future_ind])/len(self.errors_middle_dict[future_ind])
            #error_overall += sum(self.errors_overall_dict[future_ind])/len(self.errors_overall_dict[future_ind])
        self.error_middle = error_middle
        #self.error_overall = error_overall

    def deep_copy_trajectory(self):
        new_trajectory = Potential_Trajectory(self.trajectory_index, self.future_window_size, self.direction)
        new_trajectory.uncertainty = self.uncertainty
        for key, value in self.potential_cov_dict.items():
            new_trajectory.potential_cov_dict[key] = value.copy()
        new_trajectory.potential_hessian = self.potential_hessian.copy()
        for key, value in self.states.items():
            new_trajectory.states[key] = value.deep_copy_state()
        new_trajectory.inv_transformation_matrix = self.inv_transformation_matrix.clone()
        new_trajectory.drone_positions = self.drone_positions.clone()
        #new_trajectory.pitches = self.pitches
        for future_ind in range(self.future_window_size):
            new_trajectory.errors_middle_dict[future_ind] = self.errors_middle_dict[future_ind].copy()
            new_trajectory.errors_overall_dict[future_ind] = self.errors_overall_dict[future_ind].copy()
        new_trajectory.cam_list = self.cam_list.copy()
        new_trajectory.potential_2d_poses = self.potential_2d_poses.clone()
        return new_trajectory

    def print_trajectory(self):
        print("Trajectory index", self.trajectory_index)
        for key, state in self.states:
            state.print_state()
        print("Uncertainty is", self.uncertainty, "\n")

class Potential_Trajectory(object):
    def __init__(self, index, future_window_size):
        self.index = index
        self.states = {}
        self.inv_transformation_matrix = torch.zeros(future_window_size, 4, 4)
        self.drone_positions = torch.zeros(future_window_size, 3, 1)
        self.pitches = torch.zeros(future_window_size,)

    def append_to_traj(self, future_ind, potential_state):
        self.states[future_ind] = potential_state.deep_copy()
        self.inv_transformation_matrix[future_ind, :, :] = potential_state.inv_transformation_matrix
        self.drone_positions[future_ind, :, :] = potential_state.potential_C_drone
        self.pitches[future_ind] = potential_state.pitch

class PotentialStatesFetcher(object):
    def __init__(self, airsim_client, pose_client, active_parameters, loop_mode):
        self.bone_connections, self.joint_names, self.number_of_joints, self.hip_index = pose_client.model_settings()
        self.SIZE_X, self.SIZE_Y = pose_client.SIZE_X, pose_client.SIZE_Y

        self.minmax = active_parameters["MINMAX"]
        self.hessian_part = active_parameters["HESSIAN_PART"]
        self.uncertainty_calc_method = active_parameters["UNCERTAINTY_CALC_METHOD"]
        self.wobble_freq = active_parameters["WOBBLE_FREQ"]
        self.updown_lim = active_parameters["UPDOWN_LIM"]
        self.target_z_pos = active_parameters["Z_POS"]
        self.lookahead = active_parameters["LOOKAHEAD"]
        self.direction_distance = active_parameters["DIRECTION_DISTANCE"]
        self.FUTURE_WINDOW_SIZE = pose_client.FUTURE_WINDOW_SIZE
        self.ACTIVE_SAMPLING_MODE = active_parameters["ACTIVE_SAMPLING_MODE"]
        self.SAFE_RADIUS = active_parameters["SAFE_RADIUS"]
        self.TOP_SPEED =  active_parameters["TOP_SPEED"]
        self.DONT_FLY_INFRONT = active_parameters["DONT_FLY_INFRONT"]
        self.loop_mode = loop_mode
        self.movement_mode = active_parameters["MOVEMENT_MODE"]
        self.use_hessian_mode = active_parameters["USE_HESSIAN_MODE"]
        self.primary_rotation_dir = active_parameters["PRIMARY_ROTATION_DIR"]

        self.PREDEFINED_MOTION_MODE_LENGTH = pose_client.PREDEFINED_MOTION_MODE_LENGTH

        self.PRECALIBRATION_LENGTH = pose_client.PRECALIBRATION_LENGTH

        self.trajectory = active_parameters["TRAJECTORY"]
        self.is_quiet = pose_client.quiet
        self.model = pose_client.model
        self.goUp = True

        if self.primary_rotation_dir == "r":
            self.secondary_rotation_dir = ["ru", "rd"]
        elif self.primary_rotation_dir == "l":
            self.secondary_rotation_dir = ["lu", "ld"]
        self.POSITION_GRID = active_parameters["POSITION_GRID"]

        self.UPPER_LIM = active_parameters["UPPER_LIM"]
        self.LOWER_LIM = active_parameters["LOWER_LIM"]

        self.is_using_airsim = airsim_client.is_using_airsim
        self.animation = pose_client.animation
        self.already_plotted_teleport_loc = False
        self.motion_predictor = Motion_Predictor(active_parameters, self.FUTURE_WINDOW_SIZE)

        self.number_of_samples = len(self.POSITION_GRID)

        self.uncertainty_dict = {}
        self.oracle_traj_ind = None

        self.goal_state_ind = 0 
        self.goal_state = None
        self.goal_trajectory = None
        self.projection_client = pose_client.projection_client
        self.toy_example_states = []
        self.thrown_view_list = []

        if self.loop_mode == "toy_example":
            if not self.is_using_airsim:
                self.number_of_views =  airsim_client.num_of_camera_views
                self.constant_rotation_camera_sequence = airsim_client.constant_rotation_camera_sequence
                self.external_dataset_states = airsim_client.get_external_dataset_states(pose_client.modes["mode_2d"])
                self.constant_angle_view = self.constant_rotation_camera_sequence.index(airsim_client.initial_cam_view)
                self.visited_ind_index =  self.constant_angle_view 
            else:
                self.number_of_views = 18#airsim_client.num_of_camera_views
                self.constant_rotation_camera_sequence = list(np.arange(self.number_of_views ))
                self.constant_angle_view = airsim_client.initial_cam_view
                self.visited_ind_index =  self.constant_angle_view 


    def reset(self, pose_client, airsim_client, current_state):
        self.C_cam_torch = current_state.C_cam_torch
        self.current_drone_pos = np.squeeze(current_state.C_drone_gt.numpy())
        self.current_drone_vel = current_state.current_drone_vel
        
        self.current_drone_orientation = current_state.drone_orientation_gt.copy()
        self.cam_pitch = current_state.get_cam_pitch()

        self.human_GT = current_state.bone_pos_gt
        self.human_orientation_GT = current_state.human_orientation_gt
        self.human_orientation_est = current_state.human_orientation_est

        self.immediate_future_pose = pose_client.immediate_future_pose
        self.future_human_poses = torch.from_numpy(pose_client.future_poses.copy()).float()
        self.future_human_poses_gt = torch.from_numpy(pose_client.poses_3d_gt[:self.FUTURE_WINDOW_SIZE, :, :].copy()).float()

        self.current_human_pos = pose_client.current_pose

        if (pose_client.is_calibrating_energy):
            self.objective = objective_calib
        else:
            self.objective = objective_online

        if not self.is_using_airsim and not self.animation == "mpi_inf_3dhp" and self.loop_mode == "toy_example":
            self.external_dataset_states = airsim_client.get_external_dataset_states(None)

        self.immediate_future_ind = self.FUTURE_WINDOW_SIZE-1

        self.potential_trajectory_list = []
        self.thrown_trajectory_list =[]
        self.reasons_list = []
        self.toy_example_states = []
        self.potential_pose2d_list = []   
        
        self.uncertainty_dict = {}

    def restart_trajectory(self):
        self.immediate_future_ind = self.FUTURE_WINDOW_SIZE-1

    def get_potential_positions(self, linecount):
        if self.loop_mode == "teleport_simulation":
            self.get_potential_positions_sample()
        elif self.loop_mode == "flight_simulation":
            if linecount >= self.PREDEFINED_MOTION_MODE_LENGTH-1:
                self.get_potential_positions_flight()
        elif self.loop_mode == "toy_example" or self.loop_mode == "create_dataset" or self.loop_mode == "openpose_liftnet":
            self.trajectory_dome_experiment()



    def get_potential_positions_sample(self):
        #new_radius = self.SAFE_RADIUS
        unit_z = np.array([0,0,-1])

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.immediate_future_pose[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        # new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([drone_vec[1], -drone_vec[0], 0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(drone_vec)

        up_vec = np.cross(unit_horizontal, drone_vec)
        side_vec = np.cross(unit_z, drone_vec)
        up_vec_norm = up_vec*self.lookahead/np.linalg.norm(up_vec)
        side_vec_norm = side_vec*self.lookahead/np.linalg.norm(side_vec)

        use_keys =  key_indices.copy()
        use_weights = key_weights.copy()
           
        if self.ACTIVE_SAMPLING_MODE == "ellipse":
            prev_goal_index = self.goal_state_ind
            for key, value in key_indices.items():
                if prev_goal_index == value:
                    prev_goal_key = key
            use_weights = adjust_using_prev_pos(prev_goal_key, use_weights.copy())
        use_keys = remove_key_values(use_keys.copy(), current_drone_pos[2], self.LOWER_LIM, self.UPPER_LIM)
        future_weight_list = [1, 2, 3]

        for key, ind in use_keys.items():
            [weight_side, weight_up] = use_weights[key] 
            potential_trajectory = Potential_Trajectory(ind, self.FUTURE_WINDOW_SIZE, None)
            for future_pos_ind in range(0, self.FUTURE_WINDOW_SIZE):
                #future_weight = future_pos_ind + 1
                future_weight = future_weight_list[future_pos_ind]
                #future_human_loc = self.future_human_poses[self.FUTURE_WINDOW_SIZE-future_pos_ind-1, :, self.hip_index]
                #future_human_pose = self.future_human_poses[self.FUTURE_WINDOW_SIZE-future_pos_ind-1, :, :]
                pos_go = drone_vec + self.immediate_future_pose[:, self.hip_index] +  (up_vec_norm*weight_up*future_weight + side_vec_norm*future_weight*weight_side)
  
                potential_drone_vec_go = pos_go-self.immediate_future_pose[:, self.hip_index]
                norm_potential_drone_vec_go = potential_drone_vec_go * self.SAFE_RADIUS /np.linalg.norm(potential_drone_vec_go)
                go_pos_good_rad = norm_potential_drone_vec_go + self.immediate_future_pose[:, self.hip_index]

                dir_vec = go_pos_good_rad-current_drone_pos
                dir_vec_len = np.linalg.norm(dir_vec)
                desired_dist = self.lookahead
                if dir_vec_len<self.lookahead:
                    desired_dist= dir_vec_len
                norm_dir_vec = desired_dist*dir_vec/dir_vec_len
                go_pos = current_drone_pos+norm_dir_vec


                new_radius=np.linalg.norm(go_pos - self.immediate_future_pose[:, self.hip_index])
                new_theta_go = acos((go_pos[2] - self.immediate_future_pose[2, self.hip_index])/new_radius)
                new_pitch_go = pi/2 -new_theta_go
                _, new_phi_go = find_current_polar_info(current_drone_pos, self.immediate_future_pose[:, self.hip_index])
                potential_state = PotentialState(position=go_pos.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go, index=ind, C_cam_torch=self.C_cam_torch)
                pose_in_image = potential_state.get_potential_projection(self.future_human_poses[self.FUTURE_WINDOW_SIZE-future_pos_ind-1, :, :], self.projection_client)
                assert pose_in_image
                potential_trajectory.append_to_traj(future_ind=future_pos_ind, potential_state=potential_state)
                
            self.potential_trajectory_list.append(potential_trajectory)


    def get_potential_positions_flight(self):
        unit_z = np.array([0,0,-1])
        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.immediate_future_pose[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)
        radius_deviation = np.abs(cur_radius - self.SAFE_RADIUS)
        dev_tuner = 2# 9*np.exp(-4.40*radius_deviation) +1
        print("Radius deviation", radius_deviation, "dev tuner", dev_tuner)
        new_drone_vec = self.SAFE_RADIUS*(drone_vec/cur_radius)

        horizontal_comp = np.array([drone_vec[1], -drone_vec[0], 0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(drone_vec)

        up_vec = np.cross(unit_horizontal, drone_vec)
        side_vec = np.cross(unit_z, drone_vec)

        up_vec_norm = up_vec/np.linalg.norm(up_vec)
        side_vec_norm = side_vec/np.linalg.norm(side_vec)
 
        use_keys =  key_indices.copy()
        use_weights = key_weights.copy()
        # use_keys = remove_key_values(use_keys.copy(), current_drone_pos[2], self.LOWER_LIM, self.UPPER_LIM)

        for key, ind in use_keys.items():
            [weight_side, weight_up] = use_weights[key] 
            if self.movement_mode == "velocity":
                dir_vec = (up_vec_norm*weight_up + side_vec_norm*weight_side)
                norm_dir_vec = (dev_tuner*dir_vec+drone_vec) * self.SAFE_RADIUS /np.linalg.norm(dev_tuner*dir_vec+drone_vec) -drone_vec
                if np.any(norm_dir_vec):
                    if key == "c":
                        speed = 0.1
                    else:
                        speed = self.TOP_SPEED
                    desired_direction = speed * norm_dir_vec/np.linalg.norm(norm_dir_vec)
                else:
                    desired_direction = norm_dir_vec
            if self.movement_mode == "position":
                pos_go = new_drone_vec + self.immediate_future_pose[:, self.hip_index] +  self.direction_distance*(up_vec_norm*weight_up + side_vec_norm*weight_side)
                potential_drone_vec_go = pos_go-self.immediate_future_pose[:, self.hip_index]
                norm_potential_drone_vec_go = potential_drone_vec_go * self.SAFE_RADIUS /np.linalg.norm(potential_drone_vec_go)
                desired_direction = norm_potential_drone_vec_go + self.immediate_future_pose[:, self.hip_index]

            new_directions = self.motion_predictor.determine_new_direction(desired_direction)
            potential_trajectory = Potential_Trajectory(ind, self.FUTURE_WINDOW_SIZE, new_directions)
            potential_pos = self.motion_predictor.predict_potential_positions_func(desired_direction, self.current_drone_pos, self.current_drone_vel)
            for future_pos_ind in range(0, self.FUTURE_WINDOW_SIZE):
                pos_go = potential_pos[self.FUTURE_WINDOW_SIZE-future_pos_ind-1]

                temp_angle = (pos_go[2] - self.immediate_future_pose[2, self.hip_index])/np.linalg.norm(pos_go-self.immediate_future_pose[:, self.hip_index])
                if temp_angle > 1:
                    temp_angle =1
                elif temp_angle <-1:
                    temp_angle = -1
                new_theta_go = acos(temp_angle)
                new_pitch_go = pi/2 - new_theta_go
                _, new_phi_go = find_current_polar_info(current_drone_pos, self.immediate_future_pose[:, self.hip_index])

                potential_state = PotentialState(position=pos_go.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go, index=ind,  C_cam_torch=self.C_cam_torch)
                pose_in_image = potential_state.get_potential_projection(self.future_human_poses[self.FUTURE_WINDOW_SIZE-future_pos_ind-1, :, :], self.projection_client)
                #assert pose_in_image
                potential_trajectory.append_to_traj(future_ind=future_pos_ind, potential_state=potential_state)

            no_rules_broken = self.check_if_rules_broken(potential_trajectory)    
            if not no_rules_broken[0]:   
                self.potential_trajectory_list.append(potential_trajectory)
            else:
                self.thrown_trajectory_list.append(potential_trajectory)
                self.reasons_list.append(no_rules_broken)

        self.check_if_any_paths_available()
            

    # def check_if_rules_broken(self, potential_trajectory):
    #     check_future_ind = self.future_window_size-1

    #     goal_state = potential_trajectory.states[check_future_ind]
    #     drone_pos = goal_state.position
        
    #     if drone_pos[2]  > lower_lim: #about to crash
    #         reason = "hit_down"
    #         return (True, reason)

    #     if drone_pos[2]  < upper_lim:
    #         reason = "hit_up"
    #         return (True, reason)

    #     hip_pose = self.immediate_future_pose[:, self.hip_index]
    #     m1 = -np.tan(self.human_orientation_est+pi/4)
    #     m2 = -np.tan(self.human_orientation_est-pi/4)
    #     if self.DONT_FLY_INFRONT:
    #         if (drone_pos[1]-hip_pose[1]) - m1*(drone_pos[0]-hip_pose[0])>0:
    #             reason = "hit_m1"
    #             dist = np.abs(-m1*drone_pos[0]+drone_pos[1]-m1*hip_pose[0]-hip_pose[1])/np.sqrt(m1**2+1)
    #             return (True, reason, dist)
    #         if (drone_pos[1]-hip_pose[1]) - m2*(drone_pos[0]-hip_pose[0])<0: 
    #             reason = "hit_m2"
    #             dist = np.abs(-m2*drone_pos[0]+drone_pos[1]-m2*hip_pose[0]-hip_pose[1])/np.sqrt(m2**2+1)
    #             return (True, reason, dist)
    #     return (False)

    def check_if_rules_broken(self, potential_trajectory):
        check_future_ind = 0
        if self.FUTURE_WINDOW_SIZE == 1:
            check_future_ind = 0

        goal_state = potential_trajectory.states[check_future_ind]
        drone_pos = goal_state.position
        
        if self.DONT_FLY_INFRONT:
            drone_orientation = range_angle(np.arctan2(drone_pos[1], drone_pos[0]))
            upper_yaw_lim = range_angle(self.human_orientation_est + pi/4)
            lower_yaw_lim = range_angle(self.human_orientation_est - pi/4)
            print(lower_yaw_lim, drone_orientation, upper_yaw_lim)
            if (drone_orientation < upper_yaw_lim and drone_orientation > lower_yaw_lim):
                dist_lower = min(np.abs(2*pi-drone_orientation-lower_yaw_lim), np.abs(drone_orientation-lower_yaw_lim))
                dist_upper = min(np.abs(drone_orientation-upper_yaw_lim), np.abs(2*pi-drone_orientation-upper_yaw_lim))
                if dist_lower < dist_upper:
                    reason = "safety_human_right"
                    self.primary_rotation_dir = "r"
                    self.secondary_rotation_dir =["ru", "rd"]
                else:
                    reason = "safety_human_left"
                    self.primary_rotation_dir = "l"
                    self.secondary_rotation_dir =["lu", "ld"]

                dist = min(dist_lower, dist_upper)
                print(dist, reason) 
                return (True, reason, dist)

    
        if drone_pos[2]  > self.LOWER_LIM: #about to crash
            reason = "hit_down"
            dist = drone_pos[2]-self.LOWER_LIM
            return (True, reason, dist)

        if drone_pos[2]  < self.UPPER_LIM:
            reason = "hit_up"
            dist = self.UPPER_LIM-drone_pos[2]
            return (True, reason, dist)

        return (False, "all_is_well")
    
    def check_if_any_paths_available(self):
        # found_a_traj = False
        # if self.trajectory == "constant_rotation":
        #     for potential_trajectory in self.potential_trajectory_list:
        #         if potential_trajectory.trajectory_index == key_indices["r"] or potential_trajectory.trajectory_index == key_indices["l"]:
        #             found_a_traj = True
        #             break
        min_dist_safety = np.inf
        min_dist_low = np.inf
        min_dist_high = np.inf
        for ind, traj in enumerate(self.thrown_trajectory_list):
            reason_tuples = self.reasons_list[ind]
            reason = reason_tuples[1]
            distance = reason_tuples[2]
            keep_traj = None

            if (traj.trajectory_index == key_indices["r"] or traj.trajectory_index == key_indices["ru"] or traj.trajectory_index == key_indices["rd"]) and reason == "safety_human_right":
                keep_traj = traj
            elif (traj.trajectory_index == key_indices["l"] or traj.trajectory_index == key_indices["lu"] or traj.trajectory_index == key_indices["ld"]) and reason == "safety_human_left":
                keep_traj = traj
            
            if reason == "hit_up":
                if distance < min_dist_high:
                    min_dist_high = distance
                    keep_traj = traj

            if reason == "hit_down":
                if distance < min_dist_low:
                    min_dist_low = distance
                    keep_traj = traj

            if keep_traj is not None:
                self.potential_trajectory_list.append(keep_traj)
            assert len(self.potential_trajectory_list) > 0 

    def choose_trajectory(self, pose_client, linecount, online_linecount, file_manager, my_rng):
        if linecount < self.PREDEFINED_MOTION_MODE_LENGTH:
            self.choose_go_up_down()
            #self.choose_constant_rotation()
        else:
            if pose_client.is_calibrating_energy:
                if self.loop_mode == "flight_simulation" or self.loop_mode == "teleport_simulation":
                    self.choose_constant_rotation()
                elif self.loop_mode == "toy_example":
                    self.choose_constant_rotation_toy_example()
            else:
                if (self.trajectory == "active"):
                    self.find_next_state_active(pose_client, online_linecount, file_manager)  
                    file_manager.write_uncertainty_values(self.uncertainty_dict, linecount)
                    start3 = time.time()
                    self.plot_everything(linecount, file_manager, False)
                    print("plotting values took", time.time()-start3)                  
                if (self.trajectory == "constant_rotation"):
                    if self.loop_mode == "flight_simulation" or self.loop_mode == "teleport_simulation":
                        self.choose_constant_rotation()
                        # plot_potential_trajectories(self.current_human_pos, self.human_GT, self.goal_state_ind, self.potential_trajectory_list, self.hip_index, file_manager.plot_loc, linecount)            
                    elif self.loop_mode == "toy_example":
                        self.choose_constant_rotation_toy_example()
                if (self.trajectory == "random"): 
                    self.find_random_next_state(online_linecount, my_rng)
                if (self.trajectory == "constant_angle"):
                    if self.loop_mode == "flight_simulation" or self.loop_mode == "teleport_simulation":
                        self.constant_angle_baseline_future(online_linecount)
                    elif self.loop_mode == "toy_example":
                        self.choose_constant_angle_toy_example()
                if (self.trajectory == "oracle"):
                    self.choose_trajectory_using_trajind(self.oracle_traj_ind)

        if self.loop_mode == "toy_example" and not pose_client.is_calibrating_energy:
            if not self.already_plotted_teleport_loc:
                viewpoint_ind = 0
                states_dict = {}
                for theta, phi in self.POSITION_GRID:
                    new_potential_state = sample_states_spherical(self, self.SAFE_RADIUS, theta, phi, viewpoint_ind, self.C_cam_torch)
                    states_dict[viewpoint_ind] = new_potential_state
                    viewpoint_ind += 1
                plot_dome(states_dict, self.current_human_pos[:, self.hip_index], file_manager.plot_loc)
            self.already_plotted_teleport_loc = True
            file_manager.record_toy_example_results(linecount, self.potential_trajectory_list, self.uncertainty_dict, self.goal_trajectory)

        return self.goal_trajectory

    def choose_trajectory_using_trajind(self, traj_ind):
        self.goal_state_ind = traj_ind
        self.goal_trajectory = self.potential_trajectory_list[traj_ind]
        return self.goal_trajectory

    def choose_constant_rotation(self):
        not_found = True
        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.trajectory_index == key_indices[self.primary_rotation_dir]:
                self.goal_trajectory = potential_trajectory
                self.goal_state_ind = potential_trajectory.trajectory_index
                not_found=False
                print("chose", self.primary_rotation_dir)
       
        if not_found:
            for potential_trajectory in self.potential_trajectory_list:
                if potential_trajectory.trajectory_index == key_indices[self.secondary_rotation_dir[0]] or potential_trajectory.trajectory_index ==key_indices[self.secondary_rotation_dir[1]]:
                    self.goal_trajectory = potential_trajectory
                    self.goal_state_ind = potential_trajectory.trajectory_index
                    not_found=False
                    print("chose", self.secondary_rotation_dir)        

        if not_found:
            for potential_trajectory in self.potential_trajectory_list:
                if potential_trajectory.trajectory_index == key_indices["c"]:
                    self.goal_trajectory = potential_trajectory
                    self.goal_state_ind = potential_trajectory.trajectory_index
                    not_found=False
                    print("chose c")

        if not_found:
            assert NotImplementedError


    def choose_constant_rotation_toy_example(self):
        pose_in_image = False
        assert self.FUTURE_WINDOW_SIZE == 1

        while(not pose_in_image):
            self.visited_ind_index = (self.visited_ind_index+1) % len(self.constant_rotation_camera_sequence)
            if not self.is_using_airsim:
                potential_state = self.external_dataset_states[self.constant_rotation_camera_sequence[self.visited_ind_index]]
            else:
                potential_trajectory = self.potential_trajectory_list[self.constant_rotation_camera_sequence[self.visited_ind_index]]
                potential_state = potential_trajectory.states[self.FUTURE_WINDOW_SIZE-1]
            pose_in_image = potential_state.get_potential_projection(torch.from_numpy(self.immediate_future_pose).float(),  self.projection_client)

        self.goal_trajectory = Potential_Trajectory(0, self.FUTURE_WINDOW_SIZE, None)
        self.goal_trajectory.append_to_traj(future_ind=0, potential_state=potential_state)

    def choose_constant_angle_toy_example(self):
        pose_in_image = False
        ind = self.constant_angle_view
        counter = 0
        while(not pose_in_image):
            if not self.is_using_airsim:
                potential_state = self.external_dataset_states[self.constant_rotation_camera_sequence[ind]]
            else:
                if self.FUTURE_WINDOW_SIZE == 1:
                    potential_trajectory = self.potential_trajectory_list[ind]
                    potential_state = potential_trajectory.states[self.FUTURE_WINDOW_SIZE-1]
            pose_in_image = potential_state.get_potential_projection(torch.from_numpy(self.immediate_future_pose).float(),  self.projection_client)
            ind = (ind+1) % len(self.constant_rotation_camera_sequence)
            counter += 1
            if counter == len(self.constant_rotation_camera_sequence):
                potential_state = self.external_dataset_states[self.constant_rotation_camera_sequence[self.constant_angle_view]]
                break

        self.goal_trajectory = Potential_Trajectory(0, self.FUTURE_WINDOW_SIZE, None)
        self.goal_trajectory.append_to_traj(future_ind=0, potential_state=potential_state)
   

    def choose_go_up_down(self):
        current_drone_pos = self.current_drone_pos.copy()
        drone_orientation = self.current_drone_orientation.copy()

        new_radius = self.SAFE_RADIUS
        baseline_lim_up = self.UPPER_LIM
        baseline_lim_down = self.LOWER_LIM       
        
        if current_drone_pos[2] + 1 > baseline_lim_down: #about to crash
            self.goUp = True
            self.goal_state_ind = key_indices["u"]
        if self.current_drone_pos[2] -1 < baseline_lim_up:
            self.goUp = False
            self.goal_state_ind = key_indices["d"]

        if self.goUp:
            #go_dir = current_drone_pos + np.array([0,0,-self.direction_distance])
            go_dir = np.array([0,0,-self.direction_distance], dtype=np.float)
            go_pos = current_drone_pos + np.array([0,0,-0.5])
        else:
            go_dir = np.array([0,0,self.direction_distance], dtype=np.float)#current_drone_pos + np.array([0,0,self.direction_distance])
            go_pos = current_drone_pos + np.array([0,0,0.5])

        if self.movement_mode == "position":
            go_dir += current_drone_pos 

        potential_trajectory = Potential_Trajectory(0, self.FUTURE_WINDOW_SIZE, go_dir[np.newaxis].repeat(3, axis=0))
        temp_angle = (go_pos[2] - self.immediate_future_pose[2, self.hip_index])/np.linalg.norm(go_pos-self.immediate_future_pose[:, self.hip_index])
        if temp_angle > 1:
            temp_angle =1
        elif temp_angle <-1:
            temp_angle = -1
        new_theta_go = acos(temp_angle)
        new_pitch_go =pi/2 -new_theta_go
        if abs(new_pitch_go-self.cam_pitch) > pi/9:
           if self.cam_pitch > new_pitch_go:
               new_pitch_go = self.cam_pitch - pi/9
           else:
               new_pitch_go = self.cam_pitch + pi/9
   
        potential_state = PotentialState(position=go_pos.copy(), orientation=drone_orientation[2], pitch=new_pitch_go, index=self.goal_state_ind, C_cam_torch=self.C_cam_torch)
        potential_state.get_potential_projection(torch.from_numpy(self.immediate_future_pose).float(),  self.projection_client)
        potential_trajectory.append_to_traj(future_ind=0, potential_state=potential_state)
        self.goal_trajectory = potential_trajectory
        self.immediate_future_ind = self.FUTURE_WINDOW_SIZE-1

        current_radius = np.linalg.norm(current_drone_pos - self.human_GT[:, self.hip_index])
        self.SAFE_RADIUS = current_radius
        print("radius is", current_radius)


    def constant_angle_baseline_future(self, online_linecount):       
        #if online_linecount % self.FUTURE_WINDOW_SIZE == 0:
        not_found = True
        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.trajectory_index == key_indices["c"]:
                self.goal_trajectory = potential_trajectory
                not_found = False

        if not_found:
            for potential_trajectory in self.potential_trajectory_list:
                if potential_trajectory.trajectory_index == key_indices["u"] or potential_trajectory.trajectory_index == key_indices["d"]:
                    self.goal_trajectory = potential_trajectory

    def find_random_next_state(self, online_linecount, my_rng):
        #if online_linecount % self.FUTURE_WINDOW_SIZE == 0:
        random_state_ind = my_rng.get_random_state(len(self.potential_trajectory_list))
        self.goal_trajectory = self.potential_trajectory_list[random_state_ind]

    def find_next_state_active(self, pose_client, online_linecount, file_manager):
        #if online_linecount % self.FUTURE_WINDOW_SIZE == 0:
        self.find_hessians_for_potential_states(pose_client, file_manager, online_linecount)
        self.find_best_potential_state()

    def move_along_trajectory(self):
        self.goal_state = self.goal_trajectory.states[self.immediate_future_ind]
        self.immediate_future_ind -= 1
        return self.goal_state

    def prep_toy_example_trajectories(self, future_pos_ind, potential_trajectory):
        if future_pos_ind == self.FUTURE_WINDOW_SIZE:
            potential_trajectory.trajectory_index = len(self.potential_trajectory_list)
            self.potential_trajectory_list.append(potential_trajectory)

        else:
            for viewpoint_ind in range(len(self.toy_example_states)):
                new_potential_state = self.toy_example_states[viewpoint_ind]
                new_potential_state.get_potential_projection(self.future_human_poses[self.FUTURE_WINDOW_SIZE-future_pos_ind-1],  self.projection_client)
                potential_trajectory_copy = potential_trajectory.deep_copy_trajectory()    
                potential_trajectory_copy.append_to_traj(future_ind=future_pos_ind, potential_state=new_potential_state)
                self.prep_toy_example_trajectories(future_pos_ind+1, potential_trajectory_copy)

    def trajectory_dome_experiment(self):
        empty_trajectory = Potential_Trajectory(42, self.FUTURE_WINDOW_SIZE, None)
        self.toy_example_states = []
        self.potential_trajectory_list = []
        if self.is_using_airsim:
            viewpoint_ind = 0
            for theta, phi in self.POSITION_GRID:
                new_potential_state = sample_states_spherical(self, self.SAFE_RADIUS, theta, phi, viewpoint_ind, self.C_cam_torch)
                new_potential_state.get_potential_projection(torch.from_numpy(self.immediate_future_pose).float(), self.projection_client)
                self.toy_example_states.append(new_potential_state)
                viewpoint_ind += 1
        else:
            if self.animation == "mpi_inf_3dhp" or self.animation == "drone_flight":
                self.thrown_view_list = []
                for list_ind, external_dataset_state in enumerate(self.external_dataset_states):
                    pose_in_image = external_dataset_state.get_potential_projection(torch.from_numpy(self.immediate_future_pose).float(),  self.projection_client)
                    ##pose_in_image = external_dataset_state.get_potential_projection(self.future_human_poses_gt[0], self.projection_client)
                    if pose_in_image:
                        self.toy_example_states.append(external_dataset_state)
                    else:
                        self.thrown_view_list.append([list_ind, external_dataset_state.potential_projection])
                if len(self.toy_example_states) == 0:
                    self.toy_example_states = self.external_dataset_states.copy()
                print("number of states to choose from", len(self.toy_example_states))
            else: #test set for synth
                for list_ind, external_dataset_state in enumerate(self.external_dataset_states):
                    external_dataset_state.get_potential_projection(torch.from_numpy(self.immediate_future_pose).float(),  self.projection_client)
                    self.toy_example_states.append(external_dataset_state)

        self.prep_toy_example_trajectories(0, empty_trajectory)

    def find_hessians_for_potential_states(self, pose_client, file_manager, online_linecount):
        for potential_trajectory in self.potential_trajectory_list:
            self.objective.reset_hessian(pose_client, potential_trajectory, self.use_hessian_mode)
            file_manager.record_projection_est_values(pose_client.potential_projected_est, online_linecount)

            if pose_client.USE_TRAJECTORY_BASIS:
                hess2 = self.objective.hessian(pose_client.optimized_traj, self.use_hessian_mode)
            else:
                hess2 = self.objective.hessian(pose_client.optimized_poses, self.use_hessian_mode)

            potential_trajectory.set_cov(hess2, pose_client.IMMEDIATE_FUTURE_POSE_INDEX, pose_client.MIDDLE_POSE_INDEX, self.number_of_joints)
            
    def find_best_potential_state(self):
        potential_cov_lists = ["whole", "future"]
        potential_cov_dict = {}
        for potential_trajectory in self.potential_trajectory_list:
            potential_trajectory.find_uncertainty(self.uncertainty_calc_method, self.hessian_part)
            self.uncertainty_dict[potential_trajectory.trajectory_index] = potential_trajectory.uncertainty

        if (self.minmax =="use_min"):
            self.goal_state_ind = min(self.uncertainty_dict, key=self.uncertainty_dict.get)
        elif (self.minmax=="use_max"):
            self.goal_state_ind = max(self.uncertainty_dict, key=self.uncertainty_dict.get)
        #print("uncertainty list var:", np.std(uncertainty_dict.values()), "uncertainty list min max", np.min(uncertainty_dict.values()), np.max(uncertainty_dict.values()), "best ind", self.goal_state_ind)

        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.trajectory_index == self.goal_state_ind:
                self.goal_trajectory = potential_trajectory


    def plot_everything(self, linecount, file_manager, plot_potential_errors_bool):
        plot_loc = file_manager.plot_loc
        #photo_locs = file_manager.get_photo_locs()
        if not self.is_quiet:
            #plot_potential_hessians(self.potential_covs_normal, linecount, plot_loc, custom_name = "potential_covs_normal_")
            #plot_potential_hessians(self.potential_hessians_normal, linecount, plot_loc, custom_name = "potential_hess_normal_")
            #plot_potential_projections(self.potential_pose2d_list, linecount, plot_loc, photo_locs, self.bone_connections)
            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=False, top_down=False, plot_errors=True)
            #plot_potential_trajectories(self.current_human_pos, self.human_GT, self.goal_state_ind, self.potential_trajectory_list, self.hip_index, self.bone_connections, plot_loc, linecount)            
            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=True, top_down=True, plot_errors=False)
            #plot_potential_uncertainties(self, plot_loc, linecount)

            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=False, top_down=True, plot_errors=False)
            pass
            #if plot_potential_errors_bool:
            # plot_potential_errors_and_uncertainties(self, plot_loc, linecount, plot_std=False, plot_future=False, plot_log=True, custom_name="potential_errors_logplot")
            #plot_potential_errors_and_uncertainties(self, plot_loc, linecount, plot_std=False, plot_future=False, plot_log=False)
            #self.plot_projections(linecount, plot_loc)

    def plot_projections(self, linecount, plot_loc):
        plot_potential_projections_noimage(self.potential_pose2d_list, linecount, plot_loc, self.bone_connections, self.SIZE_X, self.SIZE_Y)


def remove_key_values(use_keys, current_drone_pos_z, lower_lim, upper_lim):
    if current_drone_pos_z  > lower_lim: #about to crash
        use_keys.pop("l")
        use_keys.pop("r")
        use_keys.pop("d")
        use_keys.pop("ld")
        use_keys.pop("rd")

    elif current_drone_pos_z + 1 > lower_lim: #about to crash
        use_keys.pop("d")
        use_keys.pop("ld")
        use_keys.pop("rd")

    if current_drone_pos_z  < upper_lim:
        use_keys.pop("c")
        use_keys.pop("l")
        use_keys.pop("r")
        use_keys.pop("u")
        use_keys.pop("lu")
        use_keys.pop("ru")

    elif current_drone_pos_z -1 < upper_lim:
        use_keys.pop("u")
        use_keys.pop("lu")
        use_keys.pop("ru")

    return use_keys.copy()



def adjust_using_prev_pos(prev_goal_key, use_weights):
    second_deg = 1/2
    third_deg = 1/3
    fourth_deg = 1/5
    fifth_deg = 1/8 
    if prev_goal_key == "l":
        use_weights["r"] = use_weights["r"].copy()*fifth_deg
        use_weights["u"] = use_weights["u"].copy()*third_deg
        use_weights["d"] = use_weights["d"].copy()*third_deg
        use_weights["ru"] = use_weights["ru"].copy()*fourth_deg
        use_weights["rd"] = use_weights["rd"].copy()*fourth_deg
        use_weights["lu"] = use_weights["lu"].copy()*second_deg
        use_weights["ld"] = use_weights["ld"].copy()*second_deg
    
    if prev_goal_key == "r":
        use_weights["l"] = use_weights["l"].copy()*fifth_deg
        use_weights["u"] = use_weights["u"].copy()*third_deg
        use_weights["d"] = use_weights["d"].copy()*third_deg
        use_weights["lu"] = use_weights["lu"].copy()*fourth_deg
        use_weights["ld"] = use_weights["ld"].copy()*fourth_deg
        use_weights["ru"] = use_weights["ru"].copy()*second_deg
        use_weights["rd"] = use_weights["rd"].copy()*second_deg
            
    if prev_goal_key == "u":
        use_weights["d"] = use_weights["d"].copy()*fifth_deg
        use_weights["l"] = use_weights["l"].copy()*third_deg
        use_weights["r"] = use_weights["r"].copy()*third_deg
        use_weights["lu"] = use_weights["lu"].copy()*second_deg
        use_weights["ru"] = use_weights["ru"].copy()*second_deg
        use_weights["ld"] = use_weights["ld"].copy()*fourth_deg
        use_weights["rd"] = use_weights["rd"].copy()*fourth_deg

    if prev_goal_key == "d":
        use_weights["u"] = use_weights["u"].copy()*fifth_deg
        use_weights["l"] = use_weights["l"].copy()*third_deg
        use_weights["r"] = use_weights["r"].copy()*third_deg
        use_weights["ld"] = use_weights["ld"].copy()*second_deg
        use_weights["rd"] = use_weights["rd"].copy()*second_deg
        use_weights["lu"] = use_weights["lu"].copy()*fourth_deg
        use_weights["ru"] = use_weights["ru"].copy()*fourth_deg

    if prev_goal_key == "ru":
        use_weights["ld"] = use_weights["ld"].copy()*fifth_deg
        use_weights["r"] = use_weights["r"].copy()*second_deg
        use_weights["u"] = use_weights["u"].copy()*second_deg
        use_weights["rd"] = use_weights["rd"].copy()*third_deg
        use_weights["lu"] = use_weights["lu"].copy()*third_deg
        use_weights["l"] = use_weights["l"].copy()*fourth_deg
        use_weights["d"] = use_weights["d"].copy()*fourth_deg

    if prev_goal_key == "rd":
        use_weights["lu"] = use_weights["lu"].copy()*fifth_deg
        use_weights["r"] = use_weights["r"].copy()*second_deg
        use_weights["d"] = use_weights["d"].copy()*second_deg
        use_weights["ru"] = use_weights["ru"].copy()*third_deg
        use_weights["ld"] = use_weights["ld"].copy()*third_deg
        use_weights["l"] = use_weights["l"].copy()*fourth_deg
        use_weights["u"] = use_weights["u"].copy()*fourth_deg

    if prev_goal_key == "lu":
        use_weights["rd"] = use_weights["rd"].copy()*fifth_deg
        use_weights["l"] = use_weights["l"].copy()*second_deg
        use_weights["u"] = use_weights["u"].copy()*second_deg
        use_weights["ld"] = use_weights["ld"].copy()*third_deg
        use_weights["ru"] = use_weights["ru"].copy()*third_deg
        use_weights["r"] = use_weights["r"].copy()*fourth_deg
        use_weights["d"] = use_weights["d"].copy()*fourth_deg

    if prev_goal_key == "ld":
        use_weights["ru"] = use_weights["ru"].copy()*fifth_deg
        use_weights["l"] = use_weights["l"].copy()*second_deg
        use_weights["d"] = use_weights["d"].copy()*second_deg
        use_weights["lu"] = use_weights["lu"].copy()*third_deg
        use_weights["rd"] = use_weights["rd"].copy()*third_deg
        use_weights["r"] = use_weights["r"].copy()*fourth_deg
        use_weights["u"] = use_weights["u"].copy()*fourth_deg
    
    return use_weights.copy()
