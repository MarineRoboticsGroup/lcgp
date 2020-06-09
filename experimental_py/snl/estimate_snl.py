import cvxpy as cp
import numpy as np
from numpy import linalg as la
import scipy
import copy
import time
import matplotlib.pyplot as plt

from estimate_distance_matrix import estimate_distance_matrix


def readTrajFromFile(filename ):
    trajs = []

    # open file and read the content in a list
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            traj = []

            line = line[:-1]
            # remove linebreak which is the last character of the string
            startInd = line.find('(')
            endInd = line.find(')')

            while(startInd != -1):
                sect = line[startInd+1:endInd]
                commaInd = sect.find(',')
                xcoord = float(sect[:commaInd])
                ycoord = float(sect[commaInd+1:])
                coords = (xcoord, ycoord)
                traj.append(coords)

                line = line[endInd+1:]
                startInd = line.find('(')
                endInd = line.find(')')

            trajs.append(traj)
    return trajs

def DistBetweenLocs(loc1, loc2):
    deltax = loc1[0]-loc2[0]
    deltay = loc1[1]-loc2[1]
    dist = np.sqrt(deltax**2 + deltay**2)
    return dist

def ConvertTrajsIntoConfigList(trajs):
    trajIndex = [-1 for traj in trajs]
    finalTrajIndex = [len(traj)-1 for traj in trajs]
    config_list = []
    move = []
    while not (trajIndex == finalTrajIndex):
        move.clear()
        for robotIndex in range(len(trajs)):
            # Increment trajectory for unfinished paths
            if trajIndex[robotIndex] != finalTrajIndex[robotIndex]:
                trajIndex[robotIndex] += 1
            # Get next step on paths
            newLoc = trajs[robotIndex][trajIndex[robotIndex]]
            # move += tuple(newLoc)
            move.append(tuple(newLoc))
        move_copy = copy.deepcopy(move)
        config_list.append(move_copy)
    return config_list

def CalculateAverageLocalizationError(gnd_truth, est_locs):
    n_rows, n_cols = gnd_truth.shape
    n = n_rows

    error = 0
    diff = gnd_truth - est_locs
    for row in range(n_rows):
        error += la.norm(diff[row])
    avg_error = error/float(n)
    return avg_error

def PerformSNL(loc_list):
    """
    Takes location list (list of tuples) as input
    returns average localization error
    """
    n_locs = len(loc_list)
    gnd_truth = np.array(loc_list)

    # Build Distance Matrix
    dist_matrix = np.zeros((n_locs, n_locs))
    for i, loc1 in enumerate(loc_list):
        for v2, loc2 in enumerate(loc_list[i+1:]):
            j = i+v2+1
            dist = DistBetweenLocs(loc1, loc2)
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist

    # Perform estimation w/ just SDP
    est_dist_matrix, est_locs = estimate_distance_matrix(dist_matrix, loc_list, num_anchors=3,
                                    use_model="sdp")
    sdp_error = CalculateAverageLocalizationError(gnd_truth, est_locs)

    # # Perform estimation w/ SDP and local solver
    # est_dist_matrix, est_locs = estimate_distance_matrix(dist_matrix, loc_list, num_anchors=3,
    #                                 use_model="sdp_init_spring")
    # print("SDP w/ Spring")
    # spring_error = CalculateAverageLocalizationError(gnd_truth, est_locs)
    # print()

    return sdp_error


def test():
    f1 = "rrt_traj.txt"
    f2 = "priority_traj.txt"
    files = [f1, f2]

    trajs = np.array(readTrajFromFile(f1))
    config_list = ConvertTrajsIntoConfigList(trajs)
    rrt_errors = []
    for config in config_list:
        sdp_error = PerformSNL(config)
        rrt_errors.append(sdp_error)

    trajs = np.array(readTrajFromFile(f2))
    config_list = ConvertTrajsIntoConfigList(trajs)
    our_errors = []
    for config in config_list:
        sdp_error = PerformSNL(config)
        our_errors.append(sdp_error)


    avg_rrt = sum(rrt_errors)/len(rrt_errors)
    avg_our = sum(our_errors)/len(our_errors)

    print("Avg. RRT Error:", avg_rrt)
    print("Avg. Our Error:", avg_our)

    plt.plot(rrt_errors)
    plt.plot(our_errors)
    plt.legend(["RRT", "Ours"])
    plt.show()
