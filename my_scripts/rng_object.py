import torch
import numpy as np

class rng_object(object):
    def __init__(self, seed_num):
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)

        self.rng_projection_noise = torch.get_rng_state()
        self.rng_lift_noise = torch.get_rng_state()
        self.rng_future_projection_noise = torch.get_rng_state()
        self.rng_initialization_noise = torch.get_rng_state()
        self.rng_random_traj = np.random.get_state()

        self.frozen_rng_projection_noise = self.rng_projection_noise
        self.frozen_rng_lift_noise = self.rng_lift_noise
        self.frozen_rng_future_projection_noise = self.rng_future_projection_noise
        self.frozen_rng_initialization_noise = self.rng_initialization_noise
        self.frozen_rng_random_traj = self.rng_random_traj

    def get_pose_noise(self, noise_shape, noise_std, noise_type):
        assert noise_type=="initial" or noise_type=="lift" or noise_type=="proj" or noise_type=="future_proj"
        if noise_type=="initial":
            noise_state = self.rng_initialization_noise
        elif noise_type=="lift":
            noise_state = self.rng_lift_noise 
        elif noise_type=="proj":
            noise_state = self.rng_projection_noise
        elif noise_type=="future_proj":
            noise_state = self.rng_future_projection_noise
        
        torch.set_rng_state(noise_state)
        noise = torch.normal(torch.zeros(noise_shape), torch.ones(noise_shape)*noise_std).float()
        noise_state = torch.get_rng_state()
        return noise

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

    def reload_all_rng_states(self):
        self.rng_projection_noise=self.frozen_rng_projection_noise
        self.rng_lift_noise=self.frozen_rng_lift_noise
        self.rng_future_projection_noise=self.frozen_rng_future_projection_noise
        self.rng_initialization_noise=self.frozen_rng_initialization_noise
        self.rng_random_traj=self.frozen_rng_random_traj
    