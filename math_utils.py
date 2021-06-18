import numpy as np
import warnings
import math

# from numpy import linalg as la
from scipy import linalg as la
from typing import List, Tuple
import numba
from numba import prange

# from scipy import linalg as la

eps = 1e-8

""" Testing Utils """


def assert_is_eigpair(K, eigpair):
    eigval, eigvec = eigpair
    testVec = K @ eigvec
    np.testing.assert_array_almost_equal(eigval * eigvec, testVec)
    return True


def is_square_matrix(mat):
    num_rows, num_cols = mat.shape
    return num_rows is num_cols


""" Printing Utils """


def matprint_block(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for j, x in enumerate(mat):
        if j % 2 == 0:
            print("__  __  __   __  __  __  __  __  __  __  __  __  __")
            print("")
        for i, y in enumerate(x):
            if i % 2 == 1:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end=" | ")
            else:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


""" Matrix Utils """


def get_list_all_eigvals(mat):
    assert is_square_matrix(mat)

    try:
        val = la.eigh(mat, eigvals_only=True)
        # val = la.eigvalsh(mat, eigvals_only=True, subset_by_index=[0,0])
        val[np.abs(val) < eps] = 0
        val = np.real(val)
    except:
        val = np.zeros(mat.shape[0])
    return val


def get_least_eigval(mat):
    val = la.eigvalsh(mat, subset_by_index=[0, 0])
    return float(val[0])


def get_nth_eigval(mat, n):
    """Note: n is 1-indexed

    Args:
        mat ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    index = n - 1
    try:
        val = la.eigh(mat, eigvals_only=True, subset_by_index=[index, index])
        return val
    except:
        return 0


def get_nth_eigpair(mat, n):
    assert is_square_matrix(mat)
    index = n - 1

    try:
        eigval, eigvec = la.eigh(mat, subset_by_index=[index, index])
        return eigval, eigvec
    except np.linalg.LinAlgError:
        print("Failed to converge on matrix computation")
        print(mat)
        eigvals, eigvecs = la.eig(mat)

    eigvals[np.abs(eigvals) < eps] = 0
    eigvecs = np.real(eigvecs)

    # join eigenvectors and eigenvalues for sorting
    a = np.vstack([eigvecs, eigvals])

    # take transpose to properly align
    a = a.T

    # sort array based on eigenvalues
    # least eigenvalue first
    ind = np.argsort(a[:, -1])
    sorted_eigpairs = a[ind]
    sorted_eigvals = sorted_eigpairs[:, -1:]
    sorted_eigvecs = sorted_eigpairs[:, :-1]

    desired_eigval = sorted_eigvals[index][0]
    desired_eigvec = sorted_eigvecs[index, :]
    eigpair = (desired_eigval, desired_eigvec)
    assert_is_eigpair(mat, eigpair)
    # return desired eigpair
    return eigpair


def sort_eigpairs(eigvals, eigvecs):
    """
    Sorts eigenvalues and eigenvectors from least to greatest eigenvalue and returns
    sorted arrays

    :param      eigvals:  Array of eigvals
    :type       eigvals:  np.array()
    :param      eigvecs:  Array of eigvecs
    :type       eigvecs:  np.array()

    :returns:   (sorted eigenvalues, sorted eigenvectors)
    :rtype:     (np.array(), np.array())
    """
    # join eigenvectors and eigenvalues for sorting
    a = np.vstack([eigvecs, eigvals])

    # take transpose to properly align
    a = a.T
    # sort array based on eigenvalues
    ind = np.argsort(a[:, -1])
    a = a[ind]

    # trim zero-eigenvectors and eigenvalue col
    # a = a[3:,:-1]

    srtVal = a[:, -1:].flatten()
    srtVec = a[:, :-1]
    return (srtVal, srtVec)


@numba.njit()
def build_fisher_matrix(
    edges: np.ndarray, nodes: np.ndarray, noise_model: str, noise_stddev: float
):
    """
    Stiffness matrix is actually FIM as derived in (J. Le Ny, ACC 2018)
    """
    num_nodes = len(nodes)
    num_variable_nodes = num_nodes - 3
    assert num_variable_nodes > 0
    K = np.zeros((num_variable_nodes * 2, num_variable_nodes * 2))

    alpha = None
    if noise_model == "add":
        alpha = float(1)
    elif noise_model == "lognorm":
        alpha = float(2)
    else:
        raise NotImplementedError

    std_dev_squared = noise_stddev ** 2
    for ii in range(len(edges)):
        e = edges[ii]
        i = e[0]
        j = e[1]
        if i == j:
            continue
        node_i = nodes[i]
        node_j = nodes[j]
        diff = node_i - node_j
        dist = np.linalg.norm(diff)

        # * This way of forming matrix was tested to be fastest
        denom = std_dev_squared * (dist) ** (2 * alpha)
        delX2, delY2 = np.square(diff) / denom
        delXY = diff[0] * diff[1] / denom

        i -= 3
        j -= 3

        if i >= 0:
            # Block ii
            K[2 * i, 2 * i] += delX2
            K[2 * i + 1, 2 * i + 1] += delY2
            K[2 * i, 2 * i + 1] += delXY
            K[2 * i + 1, 2 * i] += delXY

        if j >= 0:
            # Block jj
            K[2 * j, 2 * j] += delX2
            K[2 * j + 1, 2 * j + 1] += delY2
            K[2 * j, 2 * j + 1] += delXY
            K[2 * j + 1, 2 * j] += delXY

        if i >= 0 and j >= 0:
            # Block ij
            K[2 * i, 2 * j] = -delX2
            K[2 * i + 1, 2 * j + 1] = -delY2
            K[2 * i, 2 * j + 1] = -delXY
            K[2 * i + 1, 2 * j] = -delXY

            # Block ji
            K[2 * j, 2 * i] = -delX2
            K[2 * j + 1, 2 * i + 1] = -delY2
            K[2 * j, 2 * i + 1] = -delXY
            K[2 * j + 1, 2 * i] = -delXY

    # matprint_block(K)
    return K


@numba.njit()
def build_fim_from_loc_list(
    nodes_: np.ndarray, sensing_radius: float, noise_model: str, noise_stddev: float
):
    """
    Stiffness matrix is actually FIM as derived in (J. Le Ny, ACC 2018)
    """
    num_nodes = len(nodes_)
    num_variable_nodes = num_nodes - 3
    assert num_variable_nodes > 0
    K = np.zeros((num_variable_nodes * 2, num_variable_nodes * 2))

    def get_edges(nodes, sensing_radius):
        edges = []
        n_nodes = len(nodes)
        for id1 in range(n_nodes):
            for id2 in range(id1 + 1, n_nodes):
                loc1 = nodes[id1]
                loc2 = nodes[id2]
                assert loc1.size == 2
                assert loc2.size == 2
                dist = np.linalg.norm(loc1 - loc2)
                if dist < sensing_radius:
                    edges.append((id1, id2))
        return edges

    edges = get_edges(nodes_, sensing_radius)
    alpha = None
    if noise_model == "add":
        alpha = float(1)
    elif noise_model == "lognorm":
        alpha = float(2)
    else:
        raise NotImplementedError

    std_dev_squared = noise_stddev ** 2
    for ii in range(len(edges)):
        e = edges[ii]
        i = e[0]
        j = e[1]
        if i == j:
            continue
        node_i = nodes_[i]
        node_j = nodes_[j]
        diff = node_i - node_j
        dist = np.linalg.norm(diff)

        # * This way of forming matrix was tested to be fastest
        denom = std_dev_squared * (dist) ** (2 * alpha)
        delX2, delY2 = np.square(diff) / denom
        # delX2 = delXij ** 2 / denom
        # delY2 = delYij ** 2 / denom
        # delXij = diff[0]
        # delYij = diff[1]
        delXY = diff[0] * diff[1] / denom

        i -= 3
        j -= 3

        if i >= 0:
            # Block ii
            K[2 * i, 2 * i] += delX2
            K[2 * i + 1, 2 * i + 1] += delY2
            K[2 * i, 2 * i + 1] += delXY
            K[2 * i + 1, 2 * i] += delXY

        if j >= 0:
            # Block jj
            K[2 * j, 2 * j] += delX2
            K[2 * j + 1, 2 * j + 1] += delY2
            K[2 * j, 2 * j + 1] += delXY
            K[2 * j + 1, 2 * j] += delXY

        if i >= 0 and j >= 0:
            # Block ij
            K[2 * i, 2 * j] = -delX2
            K[2 * i + 1, 2 * j + 1] = -delY2
            K[2 * i, 2 * j + 1] = -delXY
            K[2 * i + 1, 2 * j] = -delXY

            # Block ji
            K[2 * j, 2 * i] = -delX2
            K[2 * j + 1, 2 * i + 1] = -delY2
            K[2 * j, 2 * i + 1] = -delXY
            K[2 * j + 1, 2 * i] = -delXY

    # matprint_block(K)
    return K


def ground_nodes_in_matrix(A, n, nodes):
    l = list(nodes)
    l.sort()
    for i in nodes:
        l.append(i + n)
    B = np.delete(A, l, axis=1)
    return B


""" Matrix Calculus """


def get_gradient_of_eigpair(K, eigpair, graph):
    """
    Returns the gradient of the eigenvalue corresponding to the eigvec and matrix
    K

    :param      K:               Given matrix
    :type       K:               np.array(2n, 2n)
    :param      eigvec:          The eigenvector
    :type       eigvec:          np.array(2n)
    :param      eigval:          corresponding eigenvalue
    :type       eigval:          float

    :returns:   Gradient of eigenvalue as function of inputs
    :rtype:     np.array(2n)

    :raises     AssertionError:  Require that given values are an eigenpair
    """

    assert_is_eigpair(K, eigpair)
    assert is_square_matrix(K)

    # pylint: disable=unused-variable
    eigval, eigvec = eigpair
    num_rows, num_cols = K.shape
    grad = np.zeros(num_rows)

    for index in range(num_rows):
        A = get_partial_deriv_of_matrix(K, index, graph)
        grad[index] = get_quadratic_multiplication(eigvec, A)

    grad = grad / la.norm(grad, 2)
    return grad


def get_partial_deriv_of_matrix(K, index, graph):
    """
    Takes partial derivative of K w.r.t. input (v) and returns partial deriv of
    matrix

    :param      K:    input Matrix to take partial deriv of
    :type       K:    np.array(2n, 2n)
    :param      i:    index of input variable (eg (i = 0, v=x0) or (i=1, v=y0))
    :type       i:    int
    """
    v = ""

    i = (int)(np.floor(index / 2))
    if index % 2 == 0:
        v = "x"
    else:
        v = "y"

    A = np.zeros_like(K)
    xi, yi = graph.get_node_loc_tuple(i)

    node_i_connections = graph.get_node_connection_list(i)

    # ensure i & j are within range of the matrix
    n = K.shape[0] / 2
    if i >= n:
        return K
    valid_connections = []
    for node in node_i_connections:
        if node < n:
            valid_connections.append(node)
    node_i_connections = valid_connections

    for j in node_i_connections:

        xj, yj = graph.get_node_loc_tuple(j)

        dKii_di = np.zeros((2, 2))
        dKjj_di = np.zeros((2, 2))
        dKij_di = np.zeros((2, 2))
        dKji_di = np.zeros((2, 2))
        if v == "x":
            dKii_di = np.array([[2 * (xi - xj), yi - yj], [yi - yj, 0]])
            dKij_di = np.array([[2 * (xj - xi), yj - yi], [yj - yi, 0]])
            dKjj_di = dKii_di
            dKji_di = dKij_di
        elif v == "y":
            dKii_di = np.array([[0, xi - xj], [xi - xj, 2 * (yi - yj)]])
            dKij_di = np.array([[0, xj - xi], [xj - xi, 2 * (yj - yi)]])
            dKjj_di = dKii_di
            dKji_di = dKij_di
        else:
            raise AssertionError

        # Kii
        A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] += dKii_di
        # Kjj
        A[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] += dKjj_di
        # Kij
        A[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += dKij_di
        # Kji
        A[2 * j : 2 * j + 2, 2 * i : 2 * i + 2] += dKji_di

    return A


""" Lin. Alg. Utils """


def calc_dist_between_locations(loc1, loc2):
    nx, ny = loc1
    gx, gy = loc2
    dx = gx - nx
    dy = gy - ny
    return math.hypot(dx, dy)


def get_quadratic_multiplication(vec, mat):
    """
    returns x.T @ A @ x

    :param      vec:  The vector
    :type       vec:  np.array(2n)
    :param      mat:  The matrix
    :type       mat:  np.array(2n, 2n)

    :returns:   resulting product in vector form
    :rtype:     np.array(2n)

    :raises     AssertionError:  x and A must be arrays of described shape
    """
    quad_result = vec.T @ mat @ vec
    return np.real(quad_result)


""" Random Generator Utils """


def generate_random_vec(nDim, length):
    vec = np.random.uniform(low=-2, high=2, size=nDim)
    vec = vec / np.linalg.norm(vec, 2)
    vec *= length
    return vec


def generate_random_tuple(lb=0, ub=10, size=2):
    vec = np.random.uniform(low=lb, high=ub, size=size)
    t = tuple(vec)
    return t


def generate_random_loc(xlb: float, xub: float, ylb: float, yub: float) -> Tuple:
    x_val = np.random.uniform(low=xlb, high=xub)
    y_val = np.random.uniform(low=ylb, high=yub)
    return (x_val, y_val)


""" Error Utils """


def calc_localization_error(gnd_truth, est_locs):
    if not (gnd_truth.shape == est_locs.shape):
        print("Ground Truth Locs", gnd_truth)
        print("Estimated Locs", est_locs)
        print(f"Gnd Truth Shape: {gnd_truth.shape}")
        print(f"Est Locs Shape: {est_locs.shape}")
        assert (
            gnd_truth.shape == est_locs.shape
        ), "The shape of the set of estimated positions did not match the expected shape"
    num_rows = gnd_truth.shape[0]
    errors = []
    diff = gnd_truth - est_locs
    for row in range(num_rows):
        errors.append(la.norm(diff[row]))
    return errors


def calc_absolute_trajectory_error(all_gnd_truths, all_est_locs):
    num_robots = all_gnd_truths[0].shape[0]
    squared_diff_sums = [0.0 for i in range(num_robots)]
    errors = [0.0 for i in range(num_robots)]

    for timestep in range(len(all_gnd_truths)):
        gnd_truth = all_gnd_truths[timestep]
        est_locs = all_est_locs[timestep]

        for robotIndex in range(num_robots):
            dist = calc_dist_between_locations(
                gnd_truth[robotIndex], est_locs[robotIndex]
            )
            squared_diff_sums[robotIndex] += dist ** 2

    print("Errors:", errors, "\nDiff sums:", squared_diff_sums)
    for robotIndex in range(num_robots):
        print(robotIndex)
        errors[robotIndex] = (squared_diff_sums[robotIndex] / num_robots) ** (1 / 2)

    return errors
