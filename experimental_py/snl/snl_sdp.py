import cvxpy as cp
import numpy as np
import scipy

import matplotlib.pyplot as plt

def GetExclusionVector(n, index):
    vec = np.zeros(n)
    vec[index] = 1
    return vec

def DistBetweenLocs(loc1, loc2):
    deltax = loc1[0]-loc2[0]
    deltay = loc1[1]-loc2[1]
    dist = np.sqrt(deltax**2 + deltay**2)
    return dist

def NodeLocsToDistanceMatrix(node_locs):
    n = len(node_locs)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            loc_i = node_locs[i]
            loc_j = node_locs[j]
            d = DistBetweenLocs(loc_i, loc_j)
            mat[i][j] = d
            mat[j][i] = d
    return mat

def DistanceMatrixToDict(dist_matrix):
    dist_dict = dict()
    n = len(dist_matrix)
    for i in range(n):
        for j in range(i+1,n):
            edge = (i,j)
            dist = dist_matrix[i,j]
            if dist > 0:
                dist_dict[edge] = dist
    return dist_dict

def SolveSNLWithSDP(num_nodes, node_node_dists, node_anchor_dists, anchor_locs, anchor_ids, init_guess=None, use_spring_solver=True):
    """
    Takes general inputs of sensor network localization problem
    and returns solved for locations. Note that right now it is
    assumed that the nodes are sorted in that there are no node ids
    greater than any anchor id. That is, ID ordering is nodes first
    then anchors

    :param      num_nodes:          Number of nodes of unknown location
    :type       num_nodes:          { type_description }
    :param      node_node_dists:    Distances between two nodes
    :type       node_node_dists:    dict{(node_id, node_id): scalar}
    :param      node_anchor_dists:  Distances between nodes and anchors
    :type       node_anchor_dists:  dict{(node_id, node_id): scalar}
    :param      anchor_locs:        Location of anchor nodes
    :type       anchor_locs:        dict{int: float}
    :param      anchor_ids:         The anchor identifiers
    :type       anchor_ids:         list of ints

    :returns:   Ordered array of estimated locations
    :rtype:     numpy.ndarray, shape = ((num_nodes+num_anchors), 2)
    """

    X = cp.Variable((2, num_nodes))
    Y = cp.Variable((num_nodes, num_nodes), symmetric=True)
    # Y = cp.Variable((num_nodes, num_nodes), PSD=True)

    # Make Z matrix
    # Leverage Schur Complement Lemma
    I2 = np.identity(2)
    Z_top = cp.hstack([I2, X])
    Z_bot = cp.hstack([X.T, Y])
    Z = cp.vstack([Z_top, Z_bot])
    constraints = [Z >> 0]

    eps_ji = dict()
    v_ji = dict()
    u_ji = dict()
    D_ji = dict()

    # Nx constraints
    for edge in node_node_dists.keys():
        i, j = edge
        # Make u_ji and v_ji variables
        u_ji[edge] = cp.Variable()
        v_ji[edge] = cp.Variable()
        # Make eps_ji a variable
        eps_ji[edge] = cp.Variable()

        D_top = cp.hstack([1, u_ji[edge]])
        D_bot = cp.hstack([u_ji[edge], v_ji[edge]])
        D_ji[edge] = cp.vstack([D_top, D_bot])
        # Constraint 1
        dist = node_node_dists[edge]
        vec = np.array([-dist, 1])
        constraints += [vec.T @ D_ji[edge] @ vec == eps_ji[edge]]

        # Constraint 3
        vec = np.zeros(num_nodes+2)
        vec[2:] = GetExclusionVector(num_nodes, i) - GetExclusionVector(num_nodes, j)
        constraints += [vec.T @ Z @ vec == v_ji[edge]]

        # Constraint 5
        constraints += [D_ji[edge] >> 0]

    eps_jk = dict()
    v_jk = dict()
    u_jk = dict()
    D_jk = dict()

    # Na constraints
    for edge in node_anchor_dists.keys():
        if edge[0] in anchor_ids and edge[1] in anchor_ids:
            continue
        elif edge[0] in anchor_ids:
            k = edge[0]
            j = edge[1]
        else:
            k = edge[1]
            j = edge[0]

        # Make u_jk and v_jk variables
        u_jk[edge] = cp.Variable()
        v_jk[edge] = cp.Variable()
        # Make eps_jk a variable
        eps_jk[edge] = cp.Variable()

        D_top = cp.hstack([1, u_jk[edge]])
        D_bot = cp.hstack([u_jk[edge], v_jk[edge]])
        D_jk[edge] = cp.vstack([D_top, D_bot])
        # Constraint 2
        dist = node_anchor_dists.get(edge)
        vec = np.array([-dist, 1])
        constraints += [vec.T @ D_jk[edge] @ vec == eps_jk[edge]]
        # Constraint 4
        vec = np.zeros(num_nodes+2)
        vec[0:2] = anchor_locs[k]
        vec[2:] = -GetExclusionVector(num_nodes, j)
        constraints += [vec.T @ Z @ vec == v_jk[edge]]
        # Constraint 6
        constraints += [D_jk[edge] >> 0]

    obj = cp.Minimize(sum(eps_jk.values())) + cp.Minimize(sum(eps_ji.values()))
    problem = cp.Problem(obj, constraints)

    if init_guess is not None:
        X.value = init_guess

    sol = problem.solve(solver=cp.SCS, warm_start=True)

    if X.value is None:
        return np.zeros((num_nodes, 2))

    locs = X.value.T

    if use_spring_solver:
        est_locs = np.array(locs)
        locs = SpringSolver(est_locs, anchor_locs, node_node_dists, node_anchor_dists)

    anchor_loc_list = np.array([])
    keys = list(anchor_locs.keys())
    keys.sort()
    for key in keys:
        locs = np.concatenate([locs, [anchor_locs[key]]])

    return locs


def SpringSolver(estimated_locs, anchor_locs, node_node_dists, node_anchor_dists):
    num_nodes = len(estimated_locs)
    n = len(estimated_locs) + len(anchor_locs)
    spring_model_params = (5*n, 0.2, 0.05, False)

    anchor_loc_arr = None
    for key in anchor_locs:
        loc = anchor_locs[key]
        if anchor_loc_arr is None:
            anchor_loc_arr = np.array([loc])
        else:
            anchor_loc_arr = np.concatenate([anchor_loc_arr, [loc]])

    max_iterations = spring_model_params[0]
    step_size = spring_model_params[1]
    epsilon = spring_model_params[2]
    show_visualization = spring_model_params[3]

    for iteration in range(max_iterations):
        sum_force = 0
        for i in range(num_nodes):
            total_force = np.array([0,0])
            for j in range(n):
                if j != i:
                    edge = (min(i,j), max(i,j))
                    if edge in node_anchor_dists:
                        dist_meas = node_anchor_dists[edge]
                        j_loc = np.array(anchor_locs[j])
                        i_loc = estimated_locs[i]
                        i_to_j = np.subtract(j_loc, i_loc)
                    elif edge in node_node_dists:
                        dist_meas = node_node_dists[edge]
                        i_to_j = np.subtract(estimated_locs[j],estimated_locs[i])
                    else:
                        continue

                    dist_est = np.linalg.norm(i_to_j)
                    e = (dist_est-dist_meas)
                    # magnitude of force applied by a pair is the error in our current estimate,
                    # weighted by how likely the RSS measurement is to be accurate
                    force = (e)*(i_to_j/dist_est)
                    total_force = np.add(total_force,force)

            estimated_locs[i] = np.add(estimated_locs[i],step_size*total_force)
            sum_force+=np.linalg.norm(total_force)

        if epsilon:
            if sum_force/n < epsilon:
                break

        if show_visualization: # visualize the algorithm's progress
            plt.scatter(estimated_locs[:,0],estimated_locs[:,1],c=list(range(num_nodes)))
            plt.scatter(anchor_loc_arr[:,0],anchor_loc_arr[:,1],c=list(range(num_nodes, n)))
            plt.pause(0.01)

    plt.clf()
    return estimated_locs

