import numpy as np
import time
import matplotlib.pyplot as plt
from snl_sdp import SolveSNLWithSDP, DistBetweenLocs, NodeLocsToDistanceMatrix, DistanceMatrixToDict

def estimate_distance_matrix(dist_matrix, loc_list, num_anchors, use_model="spring_model",spring_model_params=None):
    '''
    '''

    # number of devices
    n = dist_matrix.shape[0]
    total_num_nodes = dist_matrix.shape[0]

    dist_dict = DistanceMatrixToDict(dist_matrix)
    node_node_dists = dict()
    node_anchor_dists = dict()
    anchor_locs = dict()
    anchor_ids = []

    for id_num in range(total_num_nodes-num_anchors, total_num_nodes):
        anchor_ids.append(id_num)

    for anchor_id in anchor_ids:
        anchor_locs[anchor_id] = loc_list[anchor_id]

    for edge in dist_dict.keys():
        if edge[0] in anchor_ids and edge[1] in anchor_ids:
            continue
        elif edge[0] in anchor_ids or edge[1] in anchor_ids:
            node_anchor_dists[edge] = dist_dict[edge]
        else:
            node_node_dists[edge] = dist_dict[edge]

    n_notanchors = total_num_nodes - len(anchor_ids)
    sdp_locs = SolveSNLWithSDP(n_notanchors, node_node_dists, node_anchor_dists, anchor_locs, anchor_ids)




    # Model Dependent Section
    if use_model == "sdp":
        return np.round(NodeLocsToDistanceMatrix(sdp_locs),2), sdp_locs

    elif use_model == "sdp_init_spring":
        if spring_model_params is None:
            spring_model_params = (5*n, 0.1, 0.1, False)
        max_iterations = spring_model_params[0]
        step_size = spring_model_params[1]
        epsilon = spring_model_params[2]
        show_visualization = spring_model_params[3]

        estimated_locations = np.array(sdp_locs)
        for iteration in range(max_iterations):
            sum_force = 0
            for i in range(n):
                total_force = [0,0]
                for j in range(n):
                    if j != i:
                        i_to_j = np.subtract(estimated_locations[j],estimated_locations[i])
                        dist_est = np.linalg.norm(i_to_j)

                        dist_meas = dist_matrix[i,j]
                        e = (dist_est-dist_meas)

                        # magnitude of force applied by a pair is the error in our current estimate,
                        # weighted by how likely the RSS measurement is to be accurate
                        force = (e)*(i_to_j/dist_est)
                        total_force = np.add(total_force,force)

                estimated_locations[i] = np.add(estimated_locations[i],step_size*total_force)
                sum_force+=np.linalg.norm(total_force)

            if epsilon:
                if sum_force/n < epsilon:
                    print("\tconverged in:",iteration,"iterations")
                    break

            if show_visualization: # visualize the algorithm's progress
                plt.scatter(estimated_locations[:,0],estimated_locations[:,1],c=list(range(n)))
                plt.pause(0.01)
                time.sleep(0.01)

        # use final location estimates to populate distance matrix
        distance_matrix = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = round(np.linalg.norm(np.subtract(estimated_locations[i],estimated_locations[j])),2)
        return distance_matrix, estimated_locations

    else:
        print("use_model not defined")
        raise NotImplementedError


