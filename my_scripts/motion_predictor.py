import numpy as np

class Motion_Predictor(object):

    def __init__(self, active_parameters, future_window_size):
        acc_max_dict = active_parameters["ACC_MAX"]
        self.acc_max = np.array([acc_max_dict["x_y"], acc_max_dict["x_y"], acc_max_dict["z"]])
        self.alpha = active_parameters["ACC_ALPHA"]
        self.delta_t = active_parameters["DELTA_T"]
        self.speed = active_parameters["TOP_SPEED"]
        self.prev_acc = np.zeros((3))
        self.prev_direction = np.zeros((3))
        self.future_window_size = future_window_size
        self.direction_alpha = 0.5
        self.future_ind = self.future_window_size-1
        self.movement_mode = active_parameters["MOVEMENT_MODE"]

        if self.movement_mode == "position":
            self.predict_potential_positions_func = self.__predict_potential_positions_xgoal__


        elif self.movement_mode == "velocity":
            self.predict_potential_positions_func = self.__predict_potential_positions_vel__
            

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

    def __find_directions__(self, x_goal, x_current):
        direction = x_goal - x_current
        if np.any(direction):
            direction_unit_vector = direction / np.linalg.norm(direction)
        else: 
            direction_unit_vector = np.zeros((3,))
        return direction_unit_vector

    # def predict_potential_positions(self, x_goal, x_current, v_current):        
    #     potential_positions = np.zeros((self.future_window_size, 3))     
    #     prev_acc = self.prev_acc
    #     x_init = x_current.copy()
    #     v_init = v_current.copy()
    #     for future_ind in range(self.future_window_size-1, -1, -1):
    #         #a = self.alpha*self.acc_max*direction_unit_vector[future_ind] + (1-self.alpha)*prev_acc
    #         #prev_acc = a
    #         #if future_ind == self.future_window_size-1:
    #         #    self.prev_acc = a
    #         direction_unit_vector = self.find_directions(x_goal[future_ind], x_init)
    #         self.prev_direction = direction_unit_vector
    #         a = self.acc_max * direction_unit_vector

    #         potential_positions[future_ind, :] = x_init + v_init*self.delta_t + 0.5*a*self.delta_t**2
    #         x_init = potential_positions[future_ind, :] 
    #         v_init = v_init + a*self.delta_t
            
    #     return potential_positions


    # def determine_x_goal(self, desired_direction):
    #     new_x_goal=np.zeros([self.future_window_size,3])
    #     if self.prev_x_goal is None:
    #         self.prev_x_goal = desired_direction
    #     prev_direction = self.prev_x_goal
    #     for i in range(self.future_window_size-1, -1, -1):
    #         new_x_goal[i] = self.direction_alpha*desired_direction + (1-self.direction_alpha)*prev_direction
    #         prev_direction = new_x_goal[i] 

    #     return new_x_goal

    def determine_new_direction(self, desired_direction):
        # new_direction=np.zeros([self.future_window_size,3])
        # if self.prev_direction is None:
        #     self.prev_direction = desired_direction
        # prev_direction = self.prev_direction
        # for i in range(self.future_window_size-1, -1, -1):
        #     new_direction[i] = self.direction_alpha*desired_direction + (1-self.direction_alpha)*prev_direction
        #     prev_direction = new_direction[i] 
        new_direction = desired_direction[np.newaxis].repeat(3, axis=0)
        return new_direction

    def update_last_direction(self, direction):
        self.prev_direction = direction