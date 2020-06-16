# ActiveMoCap

The code for the CVPR 2020 paper "ActiveMoCap: Optimized Viewpoint Selection for Active Human Motion Capture" by Sena Kiciroglu, Helge Rhodin, Sudipta Sinha, Mathieu Salzmann, and Pascal Fua.

You can cite the paper as: 

    @inproceedings{kiciroglu2020activemocap,
      author = {Kiciroglu, Sena and Rhodin, Helge and Sinha, Sudipta and Salzmann, Mathieu and Fua, Pascal},
      booktitle = {CVPR},
      title = {ActiveMoCap: Optimized Viewpoint Selection for Active Human Motion Capture},
      year = {2020}
    }


## Dependencies

* numpy==1.18.1
* matplotlib==3.2.1
* torch==1.4.0
* torchvision==0.5.0
* PyYAML==5.3.1
* tornado==6.0.4
* future==0.16.0
* pandas==0.25.0

## Datasets
     
* We use the dataset [MPI_INF_3DHP] (http://gvv.mpi-inf.mpg.de/3dhp-dataset/). We have preprocessed the dataset to include only the frame range which we use, but from all the camera viewpoints used in training.  
* We have compiled our own dataset of synthetic images animated in the Unreal game engine, using the motions from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu). We use the following animations: 
    - "02_01": Subject #2, Trial #1, walk
    - "05_08": Subject #5, Trial #8, dance - rond de jambe in the air, jete, turn
    - "38_03": Subject #38, Trial #3, run around in a circle

You can access all the data used [coming soon!] (http://github.com/senakicir/ActiveMoCap).
You can evaluate the synthetic datasets and the MPI_INF_3DHP in the teleportation mode or you can load the synthetic animations into the AirSim simulation in the flight mode by specifying the name of the dataset in the config file.

You can access both datasets at *coming soon!*. 

## Trajectories

We evaluate several different baseline trajectories and our own active mode in the paper. You can specify which trajectory you would like to evaluate by specifying it in the config file. 
* "active": picks the next trajectory according to uncertainty
* "constant_rotation": picks the next trajectory according to the current travel direction.
* "random": picks a random next trajectory
* "constant_angle": picks the same viewpoint constantly

## Teleportation Mode
    
The way to replicate the experiment in the paper is the following:
1. Download the datasets from [*coming soon!*] (http://github.com/senakicir/ActiveMoCap).  
2. Specify the path to the dataset `test_sets_path` and the output path `simulation_output_path` (where you would like the results to be saved) in the config file "config_file_teleport.yaml"
3. Specify which animations you would like to evaluate in the config file, as `ANIMATION_LIST`.
4. Specify the trajectories you would like to evaluate in the config file, as `TRAJECTORY_LIST`. 
5. `cd scripts`
6. `python main.py config_file_teleport`


## Flight Mode

Instructions on how to set-up the flight mode experiment coming soon!

## Acknowledgements

We make use of the following in our pipeline:

* Openpose: https://github.com/CMU-Perceptual-Computing-Lab/openpose 
* Yolo: 
* Liftnet:
* AirSim:

We use the dataset [MPI_INF_3DHP] (http://gvv.mpi-inf.mpg.de/3dhp-dataset/), and the motions from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu)