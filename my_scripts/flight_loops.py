import numpy as np

def generate_new_goal_pos_random(curr_pos, current_direction, distance, choose_sth_random=False):
        random_int = np.random.randint(9)
        if choose_sth_random:
            random_int = 1

        if random_int == 0:
            random_goal_pos = np.random.uniform([-1,-1,-1], [1, 1, 1])
            chosen_dir = "random with up-down motion"
        elif  random_int == 1:
            new_rand_int =  np.random.randint(2)
            if new_rand_int == 0:
                random_goal_pos = np.random.uniform([-1,0, 0], [1, 0, 0])
            else:
                random_goal_pos = np.random.uniform([0,-1, 0], [0, 1, 0])
            chosen_dir = "random motion on x-y plane"
        elif  random_int == 2 or  random_int == 3 or random_int == 4 or random_int == 5 or random_int ==8:
            random_goal_pos = current_direction
            chosen_dir = "go forward in same direction"
        elif random_int == 6:
            random_goal_pos = np.zeros((3,))
            chosen_dir = "stay in current loc"
        elif random_int == 7:
            random_goal_pos = -current_direction
            chosen_dir = "go in opposite direction"

        if np.any(random_goal_pos): 
            random_goal_pos_unit = distance*random_goal_pos/np.linalg.norm(random_goal_pos)
        else:
            random_goal_pos_unit = np.zeros((3))
        #goal_pos = curr_pos + random_goal_pos_unit
        goal_pos = random_goal_pos_unit
        return goal_pos, chosen_dir

def generate_new_goal_pos_same_dir(curr_pos, direction, distance):
    if np.any(direction):
        goal_pos_unit = distance*direction/np.linalg.norm(direction)
    else:
        goal_pos_unit = np.zeros((3,))
    # goal_pos = curr_pos + goal_pos_unit
    goal_pos = goal_pos_unit
    return goal_pos
