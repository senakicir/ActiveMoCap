import numpy as np
import pdb

saved_vals_loc = "/Users/kicirogl/workspace/cvlabdata2/home/kicirogl/ActiveDrone/saved_vals"

flight_curves_dict = {}
flight_curves_loc = saved_vals_loc + "/flight_curves"
keys = ["x_curr", "v_curr", "v_final_arr", "directions", "delta_ts", "x_actual"]
for key in keys:
    flight_curves_dict[key] = np.load(flight_curves_loc+ "/"+ key + ".npy") 

x_curr = flight_curves_dict["x_curr"]
v_curr = flight_curves_dict["v_curr"]
v_final_arr = flight_curves_dict["v_final_arr"]
triple_directions = flight_curves_dict["directions"]


x_actual = flight_curves_dict["x_actual"]
delta_ts = flight_curves_dict["delta_ts"]

alphas = np.arange(1,10)*0.1
mses = []
acc_max_list = []
for alpha in alphas:
    N = delta_ts.shape[0]//3

    b_all = x_curr + v_curr* delta_ts[:,np.newaxis]

    M_all_flipped = np.zeros((N*3,3))
    directions = triple_directions[::3]
    for i in range(N):
        if i == 0:
            m_sum = alpha*directions[-1]
        else:
            m = alpha*((1-alpha)**np.arange(i))[:, np.newaxis]*directions[N-i:]
            m_sum = np.sum(m, axis=0)
        M_all_flipped[i*3:(i+1)*3,:] = m_sum

    acc_max = np.zeros((3))
    M_all =  M_all_flipped[::-1]
    for xyz_ind in range(3):
        X = x_actual[:,xyz_ind]
        b = b_all[:,xyz_ind]
        M = M_all[:, xyz_ind]
        delta_t_square = delta_ts**2
        
        acc_max[xyz_ind] = 2*(X.T - b.T)@(M*delta_t_square).T/ ((M*delta_t_square).T@(M*delta_t_square))

    print("MAX ACC SHOULD BE SET TO", acc_max, np.linalg.norm(acc_max))

    acc_max[0:2] = (acc_max[0]+acc_max[1])/2
    a = acc_max[np.newaxis]*M_all
    x_pred = b_all +  1/2*a*delta_t_square[:,np.newaxis]
    mse = np.sum((x_actual-x_pred)**2)/(N*3)
    mses.append(mse)
    acc_max_list.append(acc_max)
print("mses", mses)
print("acc_max_list", acc_max_list)