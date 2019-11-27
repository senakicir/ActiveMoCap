import numpy as np
import pdb

saved_vals_loc = "/Users/kicirogl/workspace/cvlabsrc1/home/kicirogl/ActiveDrone/saved_vals"

openpose_liftnet_sigmas = {}
openpose_liftnet_loc = saved_vals_loc + "/openpose_liftnet"
keys = ["openpose", "pose_2d_gt", "pose_lift", "pose_lift_gt"]
for key in keys:
    openpose_liftnet_sigmas[key] = np.load(openpose_liftnet_loc+ "/"+ key + ".npy") 

openpose = openpose_liftnet_sigmas["openpose"]
pose_2d_gt = openpose_liftnet_sigmas["pose_2d_gt"]
pose_lift = openpose_liftnet_sigmas["pose_lift"]
pose_lift_gt = openpose_liftnet_sigmas["pose_lift_gt"]

diff = (openpose-pose_2d_gt)
noise_mean = np.mean(diff, axis=0)
noise_std= np.std((diff), axis=0)

mask = np.logical_and(diff>noise_mean-2*noise_std, diff<noise_mean+2*noise_std)
diff = diff[np.all(mask.all(axis=1), axis=1)]

noise_mean = np.mean(diff, axis=0)
print("mean noise openpose per joint", noise_mean)

noise_std= np.std((diff), axis=0)
print("std noise openpose per joint", noise_std)

np.save(openpose_liftnet_loc + '/openpose_noise_mean', noise_mean)
np.save(openpose_liftnet_loc + '/openpose_noise_std', noise_std)

diff_lift = (pose_lift-pose_lift_gt)
noise_mean = np.mean(diff_lift, axis=0)
noise_std= np.std(diff_lift, axis=0)


mask_lift = np.logical_and(diff_lift>noise_mean-2*noise_std, diff_lift<noise_mean+2*noise_std)
mask_lift[:,:,-1] = True
diff_lift = diff_lift[np.all(mask_lift.all(axis=1), axis=1)]

noise_mean = np.mean(diff_lift, axis=0)
print("mean noise liftnet per joint", noise_mean)

noise_std= np.std(diff_lift, axis=0)
print("std noise liftnet per joint", noise_std)
np.save(openpose_liftnet_loc + '/liftnet_noise_mean', noise_mean)
np.save(openpose_liftnet_loc + '/liftnet_noise_std', noise_std)


