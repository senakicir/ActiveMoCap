import torch
import numpy as np

class rng_object(object):
    def __init__(self, seed_num, saved_vals_loc, pos_jitter_std):
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)

        self.rng_projection_noise = torch.get_rng_state()
        self.rng_lift_noise = torch.get_rng_state()
        self.rng_future_projection_noise = torch.get_rng_state()
        self.rng_initialization_noise = torch.get_rng_state()
        self.rng_random_traj = np.random.get_state()
        self.rng_pos_jitter = np.random.get_state()

        self.frozen_rng_projection_noise = self.rng_projection_noise
        self.frozen_rng_lift_noise = self.rng_lift_noise
        self.frozen_rng_future_projection_noise = self.rng_future_projection_noise
        self.frozen_rng_initialization_noise = self.rng_initialization_noise
        self.frozen_rng_random_traj = self.rng_random_traj
        self.frozen_rng_pos_jitter = self.rng_pos_jitter

        self.pose_2d_mean = torch.from_numpy(np.load(saved_vals_loc + "/openpose_liftnet/openpose_noise_mean.npy")).float()
        self.pose_2d_std = torch.from_numpy(np.load(saved_vals_loc + "/openpose_liftnet/openpose_noise_std.npy")).float()

        self.pose_lift_mean = torch.from_numpy(np.load(saved_vals_loc + "/openpose_liftnet/liftnet_noise_mean.npy")).float()
        self.pose_lift_std = torch.from_numpy(np.load(saved_vals_loc + "/openpose_liftnet/liftnet_noise_std.npy")).float()
        
        self.pos_jitter_std = pos_jitter_std

    def get_pose_noise(self, noise_shape, noise_std, noise_type):
        assert noise_type=="initial" or noise_type=="lift" or noise_type=="proj" or noise_type=="future_proj"
        if noise_type=="initial":
            noise_state = self.rng_initialization_noise
        elif noise_type=="lift":
            noise_state = self.rng_lift_noise 
            noise_mean = self.pose_lift_mean
            noise_std = self.pose_lift_std
        elif noise_type=="proj":
            noise_state = self.rng_projection_noise
            noise_mean = self.pose_2d_mean
            noise_std = self.pose_2d_std
        elif noise_type=="future_proj":
            noise_state = self.rng_future_projection_noise
            noise_mean = self.pose_2d_mean
            noise_std = self.pose_2d_std
        
        torch.set_rng_state(noise_state)
        noise = torch.normal(noise_mean, noise_std).float()
        noise_state = torch.get_rng_state()

        if noise_type=="initial":
            self.rng_initialization_noise = noise_state
        elif noise_type=="lift":
            self.rng_lift_noise = noise_state
        elif noise_type=="proj":
            self.rng_projection_noise = noise_state
        elif noise_type=="future_proj":
            self.rng_future_projection_noise = noise_state
            
        return noise

    def add_jitter(self, transformation_matrix):
        np.random.set_state(self.rng_pos_jitter)
        pos_noise = np.random.normal(loc=np.zeros(4,1), scale=np.ones(4,1)*self.pos_jitter_std)
        transformation_matrix[0:3,3] += pos_noise 
        self.rng_pos_jitter = np.random.get_state()

    def get_random_state(self, num_of_states):
        np.random.set_state(self.rng_random_traj)
        random_state_ind = np.random.randint(0, num_of_states)
        self.rng_random_traj = np.random.get_state()
        return random_state_ind

    def freeze_all_rng_states(self):
        self.frozen_rng_projection_noise = self.rng_projection_noise
        self.frozen_rng_lift_noise = self.rng_lift_noise
        self.frozen_rng_future_projection_noise = self.rng_future_projection_noise
        self.frozen_rng_initialization_noise = self.rng_initialization_noise
        self.frozen_rng_random_traj = self.rng_random_traj
        self.frozen_rng_pos_jitter = self.rng_pos_jitter

    def reload_all_rng_states(self):
        self.rng_projection_noise=self.frozen_rng_projection_noise
        self.rng_lift_noise=self.frozen_rng_lift_noise
        self.rng_future_projection_noise=self.frozen_rng_future_projection_noise
        self.rng_initialization_noise=self.frozen_rng_initialization_noise
        self.rng_random_traj=self.frozen_rng_random_traj
        self.rng_pos_jitter = self.frozen_rng_pos_jitter
    