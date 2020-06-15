import numpy as np

class Motion_Predictor(object):

    def __init__(self, active_parameters, future_window_size):
        acc_max_dict = active_parameters["ACC_MAX"]
        self.acc_max = np.array([acc_max_dict["x_y"], acc_max_dict["x_y"], acc_max_dict["z"]])
        self.alpha = active_parameters["ACC_ALPHA"]
        self.delta_t = active_parameters["DELTA_T"]
        self.speed = active_parameters["TOP_SPEED"]
        self.lookahead = active_parameters["LOOKAHEAD"]
        self.predict_accurate = active_parameters["PREDICT_ACCURATE"]
        self.prev_acc = np.zeros((3))
        self.prev_direction = np.zeros((3))
        self.future_window_size = future_window_size
        self.direction_alpha = 0.5
        self.future_ind = self.future_window_size-1
        self.movement_mode = active_parameters["MOVEMENT_MODE"]

        if self.movement_mode == "position":
            self.predict_potential_positions_func = self.__predict_potential_positions_xgoal__

        elif self.movement_mode == "velocity":
            if self.predict_accurate:
                self.predict_potential_positions_func = self.__predict_potential_positions_vel__
            else:
                self.predict_potential_positions_func = self.__predict_potential_positions_uniform__


    def __find_acc_potential_pos__(self, direction, x_current, v_current):
        a = self.alpha*self.acc_max*direction + (1-self.alpha)*self.prev_acc

        potential_positions = np.zeros((self.future_window_size, 3))            
        for future_ind in range(0, self.future_window_size):
            time_step = self.delta_t* (self.future_window_size-future_ind)
            potential_positions[future_ind, :] = x_current + v_current*time_step + 0.5*a*time_step**2
        self.prev_acc = a
        return potential_positions

    def __predict_potential_positions_xgoal__(self, x_goal, x_current, v_current):
        direction = self.__find_directions__(x_goal, x_current)
        potential_positions = self.__find_acc_potential_pos__(direction, x_current, v_current)
        return potential_positions

    def __predict_potential_positions_vel__(self, direction, x_current, v_current):
        potential_positions = self.__find_acc_potential_pos__(direction, x_current, v_current)
        return potential_positions

    def __predict_potential_positions_uniform__(self, direction, x_current, v_current):
        potential_positions = np.zeros((self.future_window_size, 3))            
        for future_ind in range(0, self.future_window_size):
            time_step = self.delta_t* (self.future_window_size-future_ind)
            potential_positions[future_ind, :] = x_current + direction*time_step*self.lookahead
        return potential_positions

    def __find_directions__(self, x_goal, x_current):
        direction = x_goal - x_current
        if np.any(direction):
            direction_unit_vector = direction / np.linalg.norm(direction)
        else: 
            direction_unit_vector = np.zeros((3,))
        return direction_unit_vector


    def determine_new_direction(self, desired_direction):
        new_direction = desired_direction[np.newaxis].repeat(3, axis=0)
        return new_direction

    def update_last_direction(self, direction):
        self.prev_direction = direction