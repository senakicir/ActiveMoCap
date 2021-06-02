import math
from time import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import maximum_filter

from .constants import *


def count_peaks(heatmap_avg, threshold):
    """
    Performs non-maximum supression on the heatmap to find the position of the joints.
    This opperation is performed channel wise, since each channel represents a different joint type.
    First a gaussian filter is applied to smooth the heatmaps.
    Afterwards each pixel in the channel is compared to its up, down, left and right neighboors and also checked that
    it's value is above the given threshold. If both of this conditions are met the point is considered a possible
    candidate to be a joint.
    The function returns for each joint type a list of candidate tuples. Each candidate tuple contains:
    [x, y, value of the original heatmap at position (x,y), counter of the candidate]
    where:
    - x is in pixles in the width direction (from left)
    - y is in pixels in the height direction (from top)

    :param heatmap_avg: The heatmap of the joints. There is one channel per joint type (C x H x W)
    :param threshold: Minimal probability when a peak is considered to be a joint
    :return: List of list containing all the joint condidates per type.
    """
    tic = time()
    all_peaks = []
    peak_counter = 0
    last_hm = len(heatmap_avg) - 1
    filter_map = np.asarray([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
        ])
    for part in range(last_hm):
        map_ori = heatmap_avg[part, ...]
        map_gf = gaussian_filter(map_ori, sigma=3)
        cond2b = map_gf > maximum_filter(map_gf, footprint=filter_map, mode='constant')
        cond1b = map_gf > threshold
        peaks_binary_new = np.logical_and(cond1b, cond2b)
        peaks = list(zip(np.nonzero(peaks_binary_new)[1], np.nonzero(peaks_binary_new)[0]))  # note reverse

        peaks_with_score_and_id = [x + (map_ori[x[1], x[0]], peak_counter + idx) for idx, x in enumerate(peaks)]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    #print "Peak counting took {}".format(time() - tic)
   # print '_' * 80
    return all_peaks


def compute_connections(paf_avg, all_peaks, threshold, image_height, model_type):
    """
    Computes the possible connections between the joint candidates and the limb candidates.

    :param paf_avg: The limbs heatmap. Is limb is represented in two channels (C x H x W)
    :param all_peaks: The joint candidates (see compute_peaks for the description of the structure)
    :param threshold: Theshold to consider or not the limb
    :param image_height: Height of the original image
    :param model_type: Defines the type of model used in the prediction (COCO or MPI)
    :return: Possible limbs found
    """
    if 'COCO' in model_type:
        mapIdx = mapIdx_COCO
        limbSeq = limbSeq_COCO
    if 'MPI' in model_type:
        mapIdx = mapIdx_MPI
        limbSeq = limbSeq_MPI

    tic = time()
    connection_all = []
    special_k = []
    mid_num = 10
    for k, map_k in enumerate(mapIdx):
        # mapIdx is a tuple so the resulting filtering has 2 channels
        score_mid = paf_avg[map_k, ...]
        candA = all_peaks[limbSeq[k][0] - 1]  # condidates for one end joint of the limb
        candB = all_peaks[limbSeq[k][1] - 1]  # candidates the other end joint of the limb
        nA = len(candA)
        nB = len(candB)
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-9  # to avoid div by 0 if same position
                    vec = np.divide(vec, norm)

                    # Choose some number of points between the two points where to verify the existance of a limb
                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[0, int(round(startend[I][1])), int(round(startend[I][0]))]
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[1, int(round(startend[I][1])), int(round(startend[I][0]))]
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * image_height / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > threshold)[0]) > 0.8 * len(score_midpts)  # Check that at least 80% of the points selected have a probability of being on the limb above the given threshold
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, float(score_with_dist_prior) + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(nA, nB):
                        break

            connection_all.append(connection)
        else:
            # At least one of the end joints of the limb has no candidates
            special_k.append(k)
            connection_all.append([])
   # print "Connection Computing took {}".format(time() - tic)
   # print '_' * 80
    return connection_all, special_k


def compute_poses(all_peaks, connection_all, special_k, model_type):
    tic = time()
    if 'COCO' in model_type:
        nb_joints = 20
        mapIdx = mapIdx_COCO
        limbSeq = limbSeq_COCO
    if 'MPI' in model_type:
        nb_joints = 17
        mapIdx = mapIdx_MPI
        limbSeq = limbSeq_MPI
    subset = -1 * np.ones((0, nb_joints))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(nb_joints)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    poses = []
    for s in subset:
        new_poses = []
        for index in s[:-2]:
            if index != -1:
                new_poses.append(candidate[int(index)][:3])
            else:
                new_poses.append([0, 0, 0])
        poses.append(np.array(new_poses))

    lip_time = time() - tic
   # print "ILP took {}".format(lip_time)
   # print '_' * 80
    return poses
