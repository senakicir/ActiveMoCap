from helpers import model_settings, choose_frame_from_cov, FUTURE_POSE_INDEX, MIDDLE_POSE_INDEX, plot_potential_ellipses, plot_potential_projections, plot_potential_hessians
import numpy as np
from State import find_current_polar_info, find_delta_yaw, SAFE_RADIUS
from determine_positions import objective_calib, objective_future
from math import radians, cos, sin, pi, degrees, acos, sqrt, inf
from random import randint
from project_bones import take_potential_projection
import time as time

TRAVEL = 0.1
TRAVEL2 = 3
UPPER_LIM = -3
LOWER_LIM = -1 #-2.5


class PotentialStatesFetcher(object):
    def __init__(self, pose_client, active_parameters):
        _, self.joint_names, _ = model_settings(pose_client.model)
        self.hip_index = self.joint_names.index('spine1')
        self.minmax = active_parameters["MINMAX"]
        self.hessian_method = active_parameters["HESSIAN_METHOD"]
        self.wobble_freq = active_parameters["WOBBLE_FREQ"]
        self.updown_lim = active_parameters["UPDOWN_LIM"]
        self.target_z_pos = active_parameters["Z_POS"]
        self.lookahead = active_parameters["LOOKAHEAD"]
        self.trajectory = active_parameters["TRAJECTORY"]
        self.is_quiet = pose_client.quiet
        self.model = pose_client.model
        self.goUp = True
        self.THETA_LIST = active_parameters["THETA_LIST"]
        self.PHI_LIST = active_parameters["PHI_LIST"]
        
        self.cv_mode_ind = [0,0]

    def reset(self, pose_client, current_state):
        self.current_drone_pos = np.squeeze(current_state.drone_pos_gt)
        self.human_GT = current_state.bone_pos_gt

        self.future_human_pos = pose_client.future_pose
        self.current_human_pos = pose_client.current_pose
        self.potential_states_try = []
        self.potential_states_go = []
        self.potential_hessians_normal = []
        self.potential_covs_normal = []

        self.current_state_ind = 0
        self.goal_state_ind =0

        self.potential_pose2d_list = []

        if (pose_client.isCalibratingEnergy):
            self.objective = objective_calib
        else:
            self.objective = objective_future

    def get_potential_positions_really_spherical_future(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = self.lookahead
        travel_go = TRAVEL2

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        side_vec = np.cross(unit_z, new_drone_vec) 
        up_vec_norm = up_vec*travel/np.linalg.norm(up_vec)
        side_vec_norm = side_vec*travel/np.linalg.norm(side_vec)
        up_vec_norm_go = up_vec*travel_go/np.linalg.norm(up_vec)
        side_vec_norm_go = side_vec*travel_go/np.linalg.norm(side_vec)

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
                self.potential_states_try.append({"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch})

                new_theta_go = acos((norm_pos_go[2] - self.future_human_pos[2, self.hip_index])/new_radius)
                new_pitch_go = pi/2 -new_theta_go
                _, new_phi_go = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
                self.potential_states_go.append({"position":np.copy(norm_pos_go), "orientation": new_phi_go+pi, "pitch": new_pitch_go})
                ind += 1

    def go_there(self, dir = "u"):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = self.lookahead
        travel_go = TRAVEL2

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        side_vec = np.cross(unit_z, new_drone_vec) 
        up_vec_norm_go = up_vec*travel_go/np.linalg.norm(up_vec)
        side_vec_norm_go = side_vec*travel_go/np.linalg.norm(side_vec)

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
        goal_state = {"position":np.copy(norm_pos_go), "orientation": new_phi_go+pi, "pitch": new_pitch_go}
        return goal_state    

    def constant_rotation_baseline_future(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = TRAVEL2

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
   
        norm_pos[2] = self.target_z_pos

        new_theta = acos((norm_pos[2] - self.future_human_pos[2, self.hip_index])/new_radius)
        new_pitch = pi/2 -new_theta
        _, new_phi = find_current_polar_info(current_drone_pos, self.future_human_pos[:, self.hip_index])
        goal_state = {"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch}

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
        goal_state = {"position":np.copy(pos_go), "orientation": new_phi_go+pi, "pitch": new_pitch_go}
        return goal_state

    #not used anymore
    def test_openpose_mode(self):
        new_radius = SAFE_RADIUS
    
        new_theta_deg = self.THETA_LIST[self.cv_mode_ind[0]]
        new_phi_deg = self.PHI_LIST[self.cv_mode_ind[1]]

        new_theta = radians(new_theta_deg)
        new_phi = radians(new_phi_deg)
        
        self.cv_mode_ind[1] += 1
        if (self.cv_mode_ind[1] == len(self.PHI_LIST)):
            self.cv_mode_ind[1] = 0
            self.cv_mode_ind[0] += 1

        #find human orientation (from GT)
        shoulder_vector = self.human_GT[:, self.joint_names.index('left_arm')] - self.human_GT[:, self.joint_names.index('right_arm')] 
        human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])

        #find coordinates of drone
        new_yaw = new_phi  + human_orientation
        x = new_radius*cos(new_yaw)*sin(new_theta) + self.human_GT[0, self.hip_index]
        y = new_radius*sin(new_yaw)*sin(new_theta) + self.human_GT[1, self.hip_index]
        z = new_radius*cos(new_theta)+ self.human_GT[2, self.hip_index]
        drone_pos = np.array([x, y, z])

        _, new_phi_go = find_current_polar_info(drone_pos, self.human_GT[:, self.hip_index]) #used to be norm_pos_go

        new_theta_go = acos((z - self.human_GT[2, self.hip_index])/new_radius)
        new_pitch = pi/2 - new_theta_go
        goal_state = {"position":drone_pos.copy(), "orientation": new_phi_go+pi, "pitch": new_pitch}
        openpose_str = str(new_theta_deg)+ "\t" + str(new_phi_deg) + "\t"
        return goal_state, openpose_str

    def dome_experiment(self):
        new_radius = SAFE_RADIUS
            
        #find human orientation (from GT)
        shoulder_vector = self.human_GT[:, self.joint_names.index('left_arm')] - self.human_GT[:, self.joint_names.index('right_arm')] 
        human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])

        for new_theta_deg in self.THETA_LIST:
            for new_phi_deg in self.PHI_LIST:
                new_theta = radians(new_theta_deg)
                new_phi = radians(new_phi_deg)

                #find coordinates of drone
                new_yaw = new_phi  + human_orientation
                x = new_radius*cos(new_yaw)*sin(new_theta) + self.human_GT[0, self.hip_index]
                y = new_radius*sin(new_yaw)*sin(new_theta) + self.human_GT[1, self.hip_index]
                z = new_radius*cos(new_theta)+ self.human_GT[2, self.hip_index]
                drone_pos = np.array([x, y, z])

                _, new_phi_go = find_current_polar_info(drone_pos, self.human_GT[:, self.hip_index]) #used to be norm_pos_go

                goal_state = {"position":np.copy(drone_pos), "orientation": new_phi_go+pi, "pitch": new_theta+pi/2}

                self.potential_states_try.append(goal_state)
                self.potential_states_go.append(goal_state)

    def up_down_baseline(self):
        new_radius = SAFE_RADIUS
        travel = TRAVEL2
        baseline_lim_up = self.updown_lim[0]
        baseline_lim_down = self.updown_lim[1]
        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        horizontal_comp = np.array([new_drone_vec[1], -new_drone_vec[0],0])
        unit_horizontal = horizontal_comp/ np.linalg.norm(new_drone_vec)

        up_vec = np.cross(unit_horizontal, new_drone_vec)
        up_vec_norm_go = up_vec*travel/np.linalg.norm(up_vec)

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
        goal_state = {"position":np.copy(norm_pos_go), "orientation": new_phi_go+pi, "pitch": new_pitch_go}
        return goal_state

    def left_right_baseline(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = TRAVEL2
        wobble_lim_up = -6
        wobble_lim_down = -2

        current_drone_pos = np.copy(self.current_drone_pos)

        drone_vec = current_drone_pos - self.future_human_pos[:, self.hip_index]
        cur_radius = np.linalg.norm(drone_vec)

        new_drone_vec = new_radius*(drone_vec/cur_radius)

        side_vec = np.cross(unit_z, new_drone_vec) 
        side_vec_norm_go = side_vec*travel/np.linalg.norm(side_vec)

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
        goal_state = {"position":np.copy(norm_pos_go), "orientation": new_phi_go+pi, "pitch": new_pitch_go}
        return goal_state

    def wobbly_baseline(self):

        new_radius = SAFE_RADIUS
        unit_z = np.array([0,0,-1])
        travel = TRAVEL2
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
        up_vec_norm_go = up_vec*travel/np.linalg.norm(up_vec)
        side_vec_norm_go = side_vec*travel/np.linalg.norm(side_vec)

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
        goal_state = {"position":np.copy(norm_pos_go), "orientation": new_phi_go+pi, "pitch": new_pitch_go}

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
        goal_state = {"position":np.copy(norm_pos), "orientation": new_phi+pi, "pitch": new_pitch}

        return goal_state
    
    def find_hessians_for_potential_states(self, pose_client, P_world):
        for potential_state_ind, potential_state in enumerate(self.potential_states_try):
            self.objective.reset_future(pose_client, potential_state)

            #start_find_hess2 = time.time()
            hess2 = self.objective.hessian(P_world)
            #end_find_hess2 = time.time()
            
            #print("Time for finding hessian no", potential_state_ind, ": ", end_find_hess2-start_find_hess2)

            self.potential_hessians_normal.append(hess2)
            inv_hess2 = np.linalg.inv(hess2)

            if (self.hessian_method == 0):
                self.potential_covs_normal.append(choose_frame_from_cov(inv_hess2, FUTURE_POSE_INDEX, self.model))
            elif (self.hessian_method == 1):
                self.potential_covs_normal.append(choose_frame_from_cov(inv_hess2, MIDDLE_POSE_INDEX, self.model))
            else:
                self.potential_covs_normal.append(inv_hess2)

            #take projection 
            #self.potential_pose2d_list.append(take_potential_projection(potential_state, self.future_human_pos)) #sloppy
        return self.potential_covs_normal, self.potential_hessians_normal

    def find_best_potential_state(self):
        uncertainty_list = []
        for cov in self.potential_covs_normal:
            if self.hessian_method == 2:
                _, s, _ = np.linalg.svd(cov)
                uncertainty_list.append(np.sum(s)) 
            else:
                _, s, _ = np.linalg.svd(cov)
                #uncertainty_list.append(np.max(s)) 
                uncertainty_list.append(np.sum(s)) 
                #uncertainty_list.append(np.linalg.det(cov))
        if (self.minmax):
            best_ind = uncertainty_list.index(min(uncertainty_list))
        else:
            best_ind = uncertainty_list.index(max(uncertainty_list))
        self.goal_state_ind = best_ind
        print("uncertainty list var:", np.std(uncertainty_list), "uncertainty list min max", np.min(uncertainty_list), np.max(uncertainty_list), "best ind", best_ind)
        goal_state = self.potential_states_go[best_ind]
        self.potential_covs_normal = []
        self.potential_hessians_normal = []
        return goal_state, best_ind

    def find_random_next_state(self):
        random_ind = randint(0, len(self.potential_states_go)-1)
        self.goal_state_ind = random_ind
        print("random ind", random_ind)
        return self.potential_states_go[random_ind]

    def plot_everything(self, linecount, plot_loc, photo_loc):
        if not self.is_quiet:
            #plot_potential_hessians(self.potential_covs_normal, linecount, plot_loc, self.model, custom_name = "potential_covs_normal_")
            #plot_potential_states(pose_client.current_pose, pose_client.future_pose, bone_pos_3d_GT, potential_states, C_drone, R_drone, pose_client.model, plot_loc, airsim_client.linecount)
            #plot_potential_projections(self.potential_pose2d_list, linecount, plot_loc, photo_loc, self.model)
            #plot_potential_ellipses(pose_client.current_pose, pose_client.future_pose, pose_client.current_pose_GT, potential_states_fetcher, pose_client.model, plot_loc_, airsim_client.linecount)
            plot_potential_ellipses(self, plot_loc, linecount, ellipses=False)
            #plot_potential_ellipses(self, plot_loc, linecount, ellipses=True)

    