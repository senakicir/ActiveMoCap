from helpers import choose_frame_from_cov,plot_potential_ellipses, plot_potential_projections, plot_potential_hessians, plot_potential_projections_noimage, euler_to_rotation_matrix, shape_cov_general, plot_potential_errors, plot_potential_errors_and_uncertainties, plot_potential_trajectories, plot_dome
import numpy as np
from State import find_current_polar_info, find_delta_yaw, SAFE_RADIUS
from determine_positions import objective_calib, objective_online
from math import radians, cos, sin, pi, degrees, acos, sqrt, inf
from project_bones import Projection_Client, C_cam_torch, CAMERA_ROLL_OFFSET, CAMERA_YAW_OFFSET, neat_tensor
import time as time
import torch

key_indices= {"c":0, "l":1, "r":2, "u":3, "d":4, "lu":5, "ld":6, "ru":7, "rd":8} 
key_weights= {"c":np.array([0,0]), "l":np.array([-1,0]), "r":np.array([1,0]), "u":np.array([0,-1]),
              "d":np.array([0,1]), "lu":np.array([-0.707,-0.707]), "ld":np.array([-0.707,0.707]), 
              "ru":np.array([0.707,-0.707]), "rd":np.array([0.707,0.707])} 


def sample_states_spherical(psf, new_radius, new_theta, new_phi, viewpoint_ind):
    new_yaw = new_phi  + psf.human_orientation_GT
    x = new_radius*cos(new_yaw)*sin(new_theta) + psf.human_GT[0, psf.hip_index]
    y = new_radius*sin(new_yaw)*sin(new_theta) + psf.human_GT[1, psf.hip_index]
    z = new_radius*cos(new_theta)+ psf.human_GT[2, psf.hip_index]
    drone_pos = np.array([x, y, z])

    _, new_phi_go = find_current_polar_info(drone_pos, psf.human_GT[:, psf.hip_index]) #used to be norm_pos_go

    a_potential_state = PotentialState(position=drone_pos.copy(), orientation=new_phi_go+pi, pitch=new_theta+pi/2, index=viewpoint_ind)
    return a_potential_state

class PotentialState(object):
    def __init__(self, position, orientation, pitch, index):
        self.position = position
        self.orientation = orientation
        self.pitch = pitch
        self.index = index

        self.potential_C_drone = torch.from_numpy(self.position[:, np.newaxis]).float()
        self.potential_R_drone = euler_to_rotation_matrix(0,0,self.orientation, returnTensor=True)
        self.potential_R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, self.pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor=True)

        drone_transformation = torch.cat((torch.cat((self.potential_R_drone, self.potential_C_drone), dim=1), neat_tensor), dim=0)
        camera_transformation = torch.cat((torch.cat((self.potential_R_cam, C_cam_torch), dim=1), neat_tensor), dim=0) 
        self.inv_transformation_matrix = torch.inverse(drone_transformation@camera_transformation)

    def get_goal_pos_yaw_pitch(self, curr_drone_orientation):
        desired_yaw_deg = find_delta_yaw((curr_drone_orientation)[2],  self.orientation)
        return self.position , desired_yaw_deg*1, self.pitch   
    
    def deep_copy_state(self):
        new_state = PotentialState(self.position, self.orientation, self.pitch, self.index)
        new_state.potential_C_drone = self.potential_C_drone.clone()
        new_state.potential_R_drone = self.potential_R_drone.clone()
        new_state.potential_R_cam = self.potential_R_cam.clone()
        new_state.inv_transformation_matrix = self.inv_transformation_matrix.clone()
        return new_state

    def print_state(self):
        print("State index:"+str(index)+". Position:", self.potential_C_drone)

class PotentialState_Drone_Flight(object):
    def __init__(self, transformation_matrix, index):
        self.transformation_matrix = transformation_matrix
        self.position = self.transformation_matrix[0:3,3]
        self.inv_transformation_matrix = torch.inverse(self.transformation_matrix)
        self.index = index
        self.pitch = 0

    def get_goal_pos_yaw_pitch(self, arg):
        return self.position, 0, 0

class Potential_Trajectory(object):
    def __init__(self, index, future_window_size):
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
        self.error_middle = 42
        self.error_overall = 42
        for future_ind in range(future_window_size):
            self.errors_middle_dict[future_ind] = []
            self.errors_overall_dict[future_ind] = []        

    def append_to_traj(self, future_ind, potential_state):
        self.states[future_ind] = potential_state.deep_copy_state()
        self.inv_transformation_matrix[self.future_window_size-future_ind-1, :, :] = potential_state.inv_transformation_matrix
        self.drone_positions[self.future_window_size-future_ind-1, :, :] = potential_state.potential_C_drone
        #self.pitches[future_ind] = potential_state.pitch

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
        self.errors_overall_dict[future_ind].append(overall_error)

    def find_overall_error(self):
        error_middle = 0
        error_overall = 0
        for future_ind in range(self.future_window_size):
            error_middle += sum(self.errors_middle_dict[future_ind])/len(self.errors_middle_dict[future_ind])
            error_overall += sum(self.errors_overall_dict[future_ind])/len(self.errors_overall_dict[future_ind])
        self.error_middle = error_middle
        self.error_overall = error_overall

    def deep_copy_trajectory(self):
        new_trajectory = Potential_Trajectory(self.trajectory_index, self.future_window_size)
        new_trajectory.uncertainty = self.uncertainty
        for key, value in self.potential_cov_dict.items():
            new_trajectory.potential_cov_dict[key] = value.copy()
        new_trajectory.potential_hessian = self.potential_hessian.copy()
        for key, value in self.states.items():
            new_trajectory.states[key] = value.deep_copy_state()
        new_trajectory.inv_transformation_matrix = self.inv_transformation_matrix.clone()
        new_trajectory.drone_positions = self.drone_positions.clone()
        #new_trajectory.pitches = self.pitches.clone()
        for future_ind in range(self.future_window_size):
            new_trajectory.errors_middle_dict[future_ind] = self.errors_middle_dict[future_ind].copy()
            new_trajectory.errors_overall_dict[future_ind] = self.errors_overall_dict[future_ind].copy()
        return new_trajectory

    def print_trajectory(self):
        print("Trajectory index", self.trajectory_index)
        for key, state in self.states:
            state.print_state()
        print("Uncertainty is", self.uncertainty, "\n")


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
        self.go_distance = active_parameters["GO_DISTANCE"]
        self.FUTURE_WINDOW_SIZE = pose_client.FUTURE_WINDOW_SIZE
        self.ACTIVE_SAMPLING_MODE = active_parameters["ACTIVE_SAMPLING_MODE"]
        self.loop_mode = loop_mode

        self.PREDEFINED_MOTION_MODE_LENGTH = pose_client.PREDEFINED_MOTION_MODE_LENGTH

        self.trajectory = active_parameters["TRAJECTORY"]
        self.is_quiet = pose_client.quiet
        self.model = pose_client.model
        self.goUp = True
        self.POSITION_GRID = active_parameters["POSITION_GRID"]

        self.UPPER_LIM = active_parameters["UPPER_LIM"]
        self.LOWER_LIM = active_parameters["LOWER_LIM"]

        self.is_using_airsim = airsim_client.is_using_airsim
        self.animation = pose_client.animation
        self.already_plotted_teleport_loc = False

        self.number_of_samples = len(self.POSITION_GRID)

        self.uncertainty_dict = {}

        self.goal_state_ind = 0 
        self.goal_state = None
        self.goal_trajectory = None
        self.visited_ind_list =[]

        if not self.is_using_airsim:
            self.drone_flight_states = airsim_client.get_drone_flight_states()
            self.number_of_samples = len(self.drone_flight_states)
    

    def reset(self, pose_client, airsim_client, current_state):
        self.current_drone_pos = np.squeeze(current_state.C_drone_gt.numpy())
        self.current_drone_orientation = current_state.drone_orientation_gt.copy()
        self.cam_pitch = current_state.cam_pitch

        self.human_GT = current_state.bone_pos_gt
        self.human_orientation_GT = current_state.human_orientation_gt

        self.future_human_pos = pose_client.immediate_future_pose
        self.current_human_pos = pose_client.current_pose

        if (pose_client.is_calibrating_energy):
            self.objective = objective_calib
        else:
            self.objective = objective_online

        self.immediate_future_ind = 0

        self.potential_trajectory_list = []
        self.potential_pose2d_list = []   
        
        self.uncertainty_dict = {}

        if not self.is_using_airsim:
            self.drone_flight_states = airsim_client.get_drone_flight_states()
            self.number_of_samples = len(self.drone_flight_states)

    def restart_trajectory(self):
        self.immediate_future_ind = 0

    def get_potential_positions(self, is_calibrating_energy):
        if self.loop_mode == "normal_simulation" or self.loop_mode == "teleport_simulation" or is_calibrating_energy:
            self.get_potential_positions_sample()
        elif self.loop_mode == "toy_example":
            self.trajectory_dome_experiment()

    def get_potential_positions_sample(self):
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

        use_keys =  key_indices.copy()
        use_weights = key_weights.copy()
           
        if self.ACTIVE_SAMPLING_MODE == "ellipse":
            prev_goal_index = self.goal_state_ind
            for key, value in key_indices.items():
                if prev_goal_index == value:
                    prev_goal_key = key
            use_weights = adjust_using_prev_pos(prev_goal_key, use_weights.copy())
        use_keys = remove_key_values(use_keys.copy(), current_drone_pos[2], self.LOWER_LIM, self.UPPER_LIM)
        future_weight_list = [1, 4, 9]

        for key, ind in use_keys.items():
            [weight_side, weight_up] = use_weights[key] 
            potential_trajectory = Potential_Trajectory(ind, self.FUTURE_WINDOW_SIZE)

            for future_pos_ind in range(0, self.FUTURE_WINDOW_SIZE):
                #future_weight = future_pos_ind + 1
                future_weight = future_weight_list[future_pos_ind]
                pos_go = new_drone_vec + self.future_human_pos[:, self.hip_index] +  (up_vec_norm_go*weight_up*future_weight + side_vec_norm_go*future_weight*weight_side)

                potential_drone_vec_go = pos_go-self.future_human_pos[:, self.hip_index]
                norm_potential_drone_vec_go = potential_drone_vec_go * new_radius /np.linalg.norm(potential_drone_vec_go)
                go_pos = norm_potential_drone_vec_go + self.future_human_pos[:, self.hip_index]

                #if key == "c" or key == "l" or key == "r":
                #    go_pos[2] = current_drone_pos[2]

                new_theta_go = acos((go_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
                new_pitch_go = pi/2 -new_theta_go
                _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
                potential_state = PotentialState(position=go_pos.copy(), orientation=new_phi_go+pi, pitch=new_pitch_go, index=ind)
                potential_trajectory.append_to_traj(future_ind=future_pos_ind, potential_state=potential_state)
                
            self.potential_trajectory_list.append(potential_trajectory)

            #debug
            #print("*******begin")
            #for potential_trajectory in self.potential_trajectory_list:
            #    print("trajectory index", potential_trajectory.index)
            #    print("trajectory transformation matrix", potential_trajectory.inv_transformation_matrix)
            #    potential_states = potential_trajectory.states
            #    for potential_state_key, potential_state in potential_states.items():
            #        print("future ind", potential_state_key)
            #        print("state index", potential_state.index)
            #        print("state position", potential_state.position)
            #        print("state orientation", potential_state.orientation)
            #        print("state pitch", potential_state.pitch)
            #        print("state inv transformation matrix",  potential_state.inv_transformation_matrix)
            #print("********end")

    def choose_trajectory(self, pose_client, linecount, online_linecount, file_manager):
        if linecount < self.PREDEFINED_MOTION_MODE_LENGTH:
            self.choose_go_up_down()
        else:
            if pose_client.is_calibrating_energy:
                self.choose_constant_rotation()
            else:
                if (self.trajectory == "active"):
                    self.find_next_state_active(pose_client, online_linecount, file_manager)                    
                    file_manager.write_uncertainty_values(self.uncertainty_dict, linecount)
                    self.plot_everything(linecount, file_manager, False)
                if (self.trajectory == "constant_rotation"):
                    if self.loop_mode == "normal_simulation" or self.loop_mode == "teleport_simulation":
                        self.choose_constant_rotation()
                    elif self.loop_mode == "toy_example":
                        print("Not implemented yet as it makes my brain hurt")
                        raise NotImplementedError
                if (self.trajectory == "random"): 
                    self.find_random_next_state(online_linecount)
                if (self.trajectory == "constant_angle"):
                    self.constant_angle_baseline_future(online_linecount)

        if self.loop_mode == "toy_example" and not pose_client.is_calibrating_energy:
            if not self.already_plotted_teleport_loc:
                viewpoint_ind = 0
                states_dict = {}
                for theta, phi in self.POSITION_GRID:
                    new_potential_state = sample_states_spherical(self, SAFE_RADIUS, theta, phi, viewpoint_ind)
                    states_dict[viewpoint_ind] = new_potential_state
                    viewpoint_ind += 1
                plot_dome(states_dict, self.current_human_pos[:, self.hip_index], file_manager.plot_loc)
            self.already_plotted_teleport_loc = True
            file_manager.record_toy_example_results(linecount, self.potential_trajectory_list, self.uncertainty_dict, self.goal_trajectory)

    def choose_trajectory_using_trajind(self, traj_ind):
        self.goal_trajectory = self.potential_trajectory_list[traj_ind]
        return self.goal_trajectory

    def choose_constant_rotation(self):
        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.trajectory_index == key_indices["r"]:
                self.goal_trajectory = potential_trajectory

    def choose_go_up_down(self):
        current_drone_pos = self.current_drone_pos.copy()
        drone_orientation = self.current_drone_orientation.copy()

        new_radius = SAFE_RADIUS
        baseline_lim_up = -3
        baseline_lim_down = -1
        
        if current_drone_pos[2] + 1 > baseline_lim_down: #about to crash
            self.goUp = True
            self.goal_state_ind = key_indices["u"]
        if current_drone_pos[2] -1 < baseline_lim_up:
            self.goUp = False
            self.goal_state_ind = key_indices["d"]

        if self.goUp:
            go_pos = current_drone_pos + np.array([0,0,-0.2])
        else:
            go_pos = current_drone_pos + np.array([0,0,0.2])

        potential_trajectory = Potential_Trajectory(0, self.FUTURE_WINDOW_SIZE)
        new_theta_go = acos((go_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch_go = self.cam_pitch #pi/2 -new_theta_go
        #if abs(new_pitch_go-self.cam_pitch) > pi/18:
        #    if self.cam_pitch > new_pitch_go:
        #        new_pitch_go = self.cam_pitch - pi/18
        #    else:
        #        new_pitch_go = self.cam_pitch + pi/18
   

        potential_state = PotentialState(position=go_pos.copy(), orientation=drone_orientation[2], pitch=new_pitch_go, index=self.goal_state_ind)
        potential_trajectory.append_to_traj(future_ind=0, potential_state=potential_state)
        self.goal_trajectory = potential_trajectory
        self.immediate_future_ind = 0

    def constant_angle_baseline_future(self, online_linecount):       
        #for potential_trajectory in self.potential_trajectory_list:
        #    if potential_trajectory.index == key_indices["c"]:
        #        goal_state = potential_trajectory[self.immediate_future_ind]
        #self.goal_state_ind = key_indices["c"]
        #if online_linecount % self.FUTURE_WINDOW_SIZE == 0:
        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.trajectory_index == key_indices["c"]:
                self.goal_trajectory = potential_trajectory

    def find_random_next_state(self, online_linecount):
        #if online_linecount % self.FUTURE_WINDOW_SIZE == 0:
        random_ind = np.random.randint(0, len(self.potential_trajectory_list)-1)
        self.goal_trajectory = self.potential_trajectory_list[random_ind]

    def find_next_state_active(self, pose_client, online_linecount, file_manager):
        #if online_linecount % self.FUTURE_WINDOW_SIZE == 0:
        self.find_hessians_for_potential_states(pose_client, file_manager, online_linecount)
        self.find_best_potential_state()

    def move_along_trajectory(self):
        self.goal_state = self.goal_trajectory.states[self.immediate_future_ind]
        self.immediate_future_ind += 1
        return self.goal_state


    #def dome_experiment(self):
    #    if self.is_using_airsim:
    #        ind = 0
    #        for theta, phi in self.POSITION_GRID:
    #            sample_states_spherical(self, SAFE_RADIUS, theta, phi, ind)
    #            ind += 1
    #    else:
    #        self.potential_states_go = self.drone_flight_states.copy()
    #    return self.potential_states_go

    def prep_theta_phi_pairs(self, future_pos_ind, potential_trajectory):
        if future_pos_ind == self.FUTURE_WINDOW_SIZE:
            potential_trajectory.trajectory_index = len( self.potential_trajectory_list)
            self.potential_trajectory_list.append(potential_trajectory)

        else:
            viewpoint_ind = 0
            for theta, phi in self.POSITION_GRID:
                new_potential_state = sample_states_spherical(self, SAFE_RADIUS, theta, phi, viewpoint_ind)
                viewpoint_ind += 1
                potential_trajectory_copy = potential_trajectory.deep_copy_trajectory()    
                potential_trajectory_copy.append_to_traj(future_ind=future_pos_ind, potential_state=new_potential_state)
                self.prep_theta_phi_pairs(future_pos_ind+1, potential_trajectory_copy)

    def trajectory_dome_experiment(self):
        if self.is_using_airsim:
            future_pos_ind = 0 
            potential_trajectory = Potential_Trajectory(42, self.FUTURE_WINDOW_SIZE)
            self.prep_theta_phi_pairs(future_pos_ind, potential_trajectory)
        else:
            print("Not implemented yet like all other drone data stuff")
            raise NotImplementedError

    def find_hessians_for_potential_states(self, pose_client, file_manager, online_linecount):
        for potential_trajectory in self.potential_trajectory_list:
            self.objective.reset_future(pose_client, potential_trajectory)
            file_manager.record_projection_est_values(pose_client.potential_projected_est, online_linecount)

            if pose_client.USE_TRAJECTORY_BASIS:
                hess2 = self.objective.hessian(pose_client.optimized_traj)
            else:
                hess2 = self.objective.hessian(pose_client.optimized_poses)

            potential_trajectory.set_cov(hess2, pose_client.FUTURE_POSE_INDEX, pose_client.MIDDLE_POSE_INDEX, self.number_of_joints)
            #future_pose = torch.from_numpy(self.future_human_pos).float() 
            #self.potential_pose2d_list.append(pose_client.projection_client.take_single_projection(future_pose, potential_state.inv_transformation_matrix))

    def find_best_potential_state(self):
        potential_cov_lists = ["whole", "future"]
        potential_cov_dict = {}
        for potential_trajectory in self.potential_trajectory_list:
            potential_trajectory.find_uncertainty(self.uncertainty_calc_method, self.hessian_part)
            self.uncertainty_dict[potential_trajectory.trajectory_index] = potential_trajectory.uncertainty

        if (self.minmax):
            self.goal_state_ind = min(self.uncertainty_dict, key=self.uncertainty_dict.get)
        else:
            self.goal_state_ind = max(self.uncertainty_dict, key=self.uncertainty_dict.get)
        #print("uncertainty list var:", np.std(uncertainty_dict.values()), "uncertainty list min max", np.min(uncertainty_dict.values()), np.max(uncertainty_dict.values()), "best ind", self.goal_state_ind)

        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.trajectory_index == self.goal_state_ind:
                self.goal_trajectory = potential_trajectory

    def find_next_state_constant_rotation(self, linecount):
        if self.animation != "drone_flight":
            self.goal_state_ind = (linecount)%(len(self.potential_states_go))
            for potential_states in self.potential_states_go:
                if potential_states.index == self.goal_state_ind:
                    self.goal_state = potential_states
        else:
            prev_goal_state_ind = self.goal_state_ind
            for potential_states in self.potential_states_go:
                if potential_states.index == prev_goal_state_ind:
                    prev_goal_state_position = potential_states.position
            
            min_dist = 1000000
            for potential_states in self.potential_states_go:
                next_position = potential_states.position
                dist_to_next_position = torch.norm(next_position-prev_goal_state_position)
                if potential_states.index not in self.visited_ind_list:
                    if  min_dist > dist_to_next_position:
                        min_dist = dist_to_next_position
                        self.goal_state = potential_states
                        self.goal_state_ind = self.goal_state.index

            if len(self.visited_ind_list) == len(self.potential_states_go)-5:
                self.visited_ind_list.pop(0)
            self.visited_ind_list.append(self.goal_state_ind)

        return self.goal_state

    def choose_state(self, index, future_step):
        self.goal_state_ind = index
        for potential_trajectory in self.potential_trajectory_list:
            if potential_trajectory.index == self.goal_state_ind:
                self.goal_state = potential_trajectory.states[self.immediate_future_ind]
        return self.goal_state


    def plot_everything(self, linecount, file_manager, plot_potential_errors_bool):
        plot_loc = file_manager.plot_loc
        #photo_locs = file_manager.get_photo_locs()
        if not self.is_quiet:
            #plot_potential_hessians(self.potential_covs_normal, linecount, plot_loc, custom_name = "potential_covs_normal_")
            #plot_potential_hessians(self.potential_hessians_normal, linecount, plot_loc, custom_name = "potential_hess_normal_")
            #plot_potential_projections(self.potential_pose2d_list, linecount, plot_loc, photo_locs, self.bone_connections)
            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=False, top_down=False, plot_errors=True)
            plot_potential_trajectories(self.current_human_pos, self.human_GT, self.goal_state_ind, self.potential_trajectory_list, self.hip_index, plot_loc, linecount)            
            plot_potential_ellipses(self, plot_loc, linecount, ellipses=True, top_down=True, plot_errors=False)
            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=False, top_down=True, plot_errors=False)

            #if plot_potential_errors_bool:
              #  plot_potential_errors_and_uncertainties(self, plot_loc, linecount, plot_std=False, plot_future=False, plot_log=True, custom_name="potential_errors_logplot")
             #   plot_potential_errors_and_uncertainties(self, plot_loc, linecount, plot_std=False, plot_future=False, plot_log=False)
            #self.plot_projections(linecount, plot_loc)

    def plot_projections(self, linecount, plot_loc):
        plot_potential_projections_noimage(self.potential_pose2d_list, linecount, plot_loc, self.bone_connections, self.SIZE_X, self.SIZE_Y)


def remove_key_values(use_keys, current_drone_pos_z, lower_lim, upper_lim):
    if current_drone_pos_z  > lower_lim: #about to crash
        use_keys.pop("c")
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