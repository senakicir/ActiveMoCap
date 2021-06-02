import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity



#base_experiment_loc = '/cvlabsrc1/home/kicirogl/ActiveDrone/simulation_results/experiments_2020-02-04-17-07_b_/2020-02-04-17-07/05_08/'
oracle_1_loc = '/cvlabsrc1/home/kicirogl/ActiveDrone/simulation_results/experiments_2020-02-05-17-29/2020-02-05-17-29/mpi_inf_3dhp/'
#base_experiment_loc = '/cvlabsrc1/home/kicirogl/ActiveDrone/simulation_results/experiments_2020-02-05-17-32/2020-02-05-17-32/02_01/'
#oracle_1_loc = '/cvlabsrc1/home/kicirogl/ActiveDrone/simulation_results/experiments_2020-02-07-23-52/2020-02-07-23-52/mpi_inf_3dhp/'
oracle_2_loc = '/cvlabsrc1/home/kicirogl/ActiveDrone/simulation_results/experiments_2020-02-07-23-53/2020-02-07-23-53/mpi_inf_3dhp/'


pearson_corr_list = []
diff_in_error_list = []
cosine_sim_list = []
top_3_count = []
top_5_count = []
top_5_count_rand = []
top_3_count_rand = []

threshold = 0.05
seeds = [5,41,3,10,12]

for folder_num in range(5):
    oracle_1_loc_exp = oracle_1_loc+ str(folder_num) + '/'
    oracle_2_loc_exp = oracle_2_loc+ str(folder_num) + '/'

    error_loc = oracle_1_loc_exp+'oracle_errors.txt'
    uncertainty_loc = oracle_2_loc_exp+'oracle_errors.txt'

    f_error_values=open(error_loc, "r")
    f_uncertainty_values = open(uncertainty_loc, "r")
    error_values = f_error_values.readlines()
    uncertainty_values = f_uncertainty_values.readlines()
    np.random.seed(seeds[folder_num])
    for i in range(len(uncertainty_values)-1):
        errors = error_values[i+1].split("\t")
        errors = errors[1:-1] 
        uncertainties = uncertainty_values[i+1].split("\t")
        uncertainties = uncertainties[1:-1] 

        errors = [float(error) for error in errors]
        uncertainties = [float(uncert) for uncert in uncertainties]
        uncertainties=np.array(uncertainties)
        errors=np.array(errors)

        # print(uncertainties)
        #errors = (errors-np.mean(errors))/np.std(errors)
        #uncertainties = np.power(uncertainties, 1/6)
        #uncertainties = (uncertainties-np.mean(uncertainties))/np.std(uncertainties)

        assert (len(uncertainties)==len(errors))
        corr = pearsonr(uncertainties, errors)[0]

        diff_in_error = np.max(errors) - np.min(errors)
        cos_sim = cosine_similarity(uncertainties[np.newaxis], errors[np.newaxis])
       # print(corr)

        min_uncert_ind = np.argmin(uncertainties)
        rand_ind = np.random.randint(0, len(uncertainties))
        top_5_error_ind = np.argsort(errors)[:5]
        top_3_error_ind = np.argsort(errors)[:3]

        if diff_in_error > threshold:
            pearson_corr_list.append(corr)
            cosine_sim_list.append(cos_sim)
            diff_in_error_list.append(diff_in_error)

            if min_uncert_ind in top_5_error_ind:
                top_5_count.append(1)
            else:
                top_5_count.append(0)

            if rand_ind in top_5_error_ind:
                top_5_count_rand.append(1)
            else:
                top_5_count_rand.append(0)

            if min_uncert_ind in top_3_error_ind:
                top_3_count.append(1)
            else:
                top_3_count.append(0)

            if rand_ind in top_3_error_ind:
                top_3_count_rand.append(1)
            else:
                top_3_count_rand.append(0)

#print(diff_in_error_list)
pearson_corr_arr = np.array(pearson_corr_list)
diff_in_error_arry = np.array(diff_in_error_list)
cosine_sim_arry = np.array(cosine_sim_list)

print("num of samples matter", len(diff_in_error_list))
print("pearson_corr_arr", np.mean(pearson_corr_arr), np.std(pearson_corr_arr))
#print("cosine sim", np.mean(cosine_sim_arry), np.std(cosine_sim_arry))

print("top 5 perc:", sum(top_5_count)/len(top_5_count)*100)
print("top 5 perc, random:", sum(top_5_count_rand)/len(top_5_count_rand)*100)
print("top 3 perc:", sum(top_3_count)/len(top_3_count)*100)
print("top 3 perc, random:", sum(top_3_count_rand)/len(top_3_count_rand)*100)
