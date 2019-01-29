import numpy as np

class FileManager(object):
    def __init__(self, parameters):
        self.anim_num = parameters["ANIMATION_NUM"]
        self.experiment_name = parameters["EXPERIMENT_NAME"]
        self.test_set_name = parameters["TEST_SET_NAME"]

        self.file_names = parameters["FILE_NAMES"]
        self.folder_names = parameters["FOLDER_NAMES"]


        self.foldernames_anim = self.folder_names[self.experiment_name]
        self.filenames_anim = self.file_names[self.experiment_name]

        #open files
        self.f_output = open(self.filenames_anim["f_output"], 'w')
        self.f_groundtruth = open(self.filenames_anim["f_groundtruth"], 'w')
        self.f_reconstruction = open(self.filenames_anim["f_reconstruction"], 'w')
        self.f_openpose_error = open(self.filenames_anim["f_openpose_error"], 'w')

        self.plot_loc = self.foldernames_anim["superimposed_images"]
        self.take_photo_loc =  self.foldernames_anim["images"]
        self.estimate_folder_name = self.foldernames_anim["estimates"]

        self.openpose_err_str = ""

    def get_nonairsim_client_names(self):
        return 'test_sets/'+self.test_set_name+'/groundtruth.txt', 'test_sets/'+self.test_set_name+'/a_flight.txt'

    def save_simulation_values(self, airsim_client, pose_client):
        f_output_str = str(airsim_client.linecount)+pose_client.f_string + '\n'
        self.f_output.write(f_output_str)

        f_reconstruction_str = str(airsim_client.linecount)+ '\t' + pose_client.f_reconst_string + '\n'
        self.f_reconstruction.write(f_reconstruction_str)

        f_groundtruth_str =  str(airsim_client.linecount) + '\t' + pose_client.f_groundtruth_str + '\n'
        self.f_groundtruth.write(f_groundtruth_str)

    def save_initial_drone_pos(self, airsim_client):
        f_groundtruth_prefix = "-1\t" + str(airsim_client.DRONE_INITIAL_POS[0,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[1,]) + "\t" + str(airsim_client.DRONE_INITIAL_POS[2,])
        for i in range(0,70):
            f_groundtruth_prefix = f_groundtruth_prefix + "\t"
        self.f_groundtruth.write(f_groundtruth_prefix + "\n")

    def get_photo_loc(self, linecount, USE_AIRSIM):
        if (USE_AIRSIM):
            photo_loc = self.foldernames_anim["images"] + '/img_' + str(linecount) + '.png'
        else:
            photo_loc = 'test_sets/'+self.test_set_name+'/images/img_' + str(linecount) + '.png'
        return photo_loc
    
    def close_files(self):
        self.f_groundtruth.close()
        self.f_output.close()
        self.f_reconstruction.close()
        self.f_openpose_error.close()

    def write_openpose_prefix(self,THETA_LIST, PHI_LIST, num_of_joints):
        prefix_string = ""
        for new_theta_deg in THETA_LIST:
            for new_phi_deg in PHI_LIST:
                prefix_string += str(new_theta_deg) +", " + str(new_phi_deg) + '\t'
        
        for _ in range(num_of_joints):
            prefix_string += '\t'
        self.f_openpose_error.write(prefix_string + "\n")

    def append_openpose_error(self, err):
        self.openpose_err_str += str(err) + "\t"

    def write_openpose_error(self, human_pose):
        for i in range(human_pose.shape[1]):
            self.openpose_err_str += str(human_pose[0,i]) + "\t" + str(human_pose[1,i]) + "\t" + str(human_pose[2,i]) + "\t"

        self.f_openpose_error.write(self.openpose_err_str + "\n")
        self.openpose_err_str = ""