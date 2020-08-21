import numpy as np
from numpy import linalg as la
import graph

eps = 1e-8

####### Testing Utils #######
def assertIsEigpair(K, eigval, eigvect):
    testVec = K @ eigvect
    np.testing.assert_array_almost_equal(eigval * eigvect, testVec)
    return True

def assertIsEigpair(K, eigpair):
    eigval, eigvect = eigpair
    testVec = K @ eigvect
    np.testing.assert_array_almost_equal(eigval * eigvect, testVec)
    return True

def isSquareMatrix(mat):
    nrow, ncol = mat.shape
    return (nrow is ncol)

####### Printing Utils #######
def matprintBlock(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for j, x in enumerate(mat):
        if j%2 == 0:
            print("__  __  __   __  __  __  __  __  __  __  __  __  __")
            print("")
        for i, y in enumerate(x):
            if i%2 == 1:
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end=" | ")
            else:
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

####### Matrix Utils #######
def getListOfAllEigvals(mat):
    assert(isSquareMatrix(mat))

    val, vect = la.eig(mat)
    val[np.abs(val) < eps] = 0
    val = np.real(val)
    return val

def getNthEigpair(mat, n):
    assert(isSquareMatrix(mat))
    index = n-1

    eigvals, eigvects = la.eig(mat)
    eigvals[np.abs(eigvals) < eps] = 0
    eigvects = np.real(eigvects)

    # join eigenvects and vals for sorting
    a = np.vstack([eigvects, eigvals])

    # take transpose to properly align
    a = a.T

    # sort array based on eigenvalues
    # least eigenvalue first
    ind=np.argsort(a[:,-1])
    sortedEigenpairs=a[ind]
    sortedEigenvals = sortedEigenpairs[:,-1:]
    sortedEigenvects = sortedEigenpairs[:,:-1]

    indexEigVal = sortedEigenvals[index][0]
    indexEigVect = sortedEigenvects[index,:]
    eigpair = (indexEigVal, indexEigVect)
    assertIsEigpair(mat, eigpair)
    # return desired eigenpair
    return eigpair

def sortEigs(vals, vecs):
    """
    Sorts eigenvals and eigenvecs from least to greatest eigenval and returns
    sorted arrays

    :param      vals:  Array of eigenvals
    :type       vals:  np.array()
    :param      vecs:  Array of eigenvecs
    :type       vecs:  np.array()

    :returns:   (sorted eigenvals, sorted eigenvects)
    :rtype:     (np.array(), np.array())
    """
    # join eigenvects and vals for sorting
    a = np.vstack([vecs, vals])

    # take transpose to properly align
    a = a.T
    # sort array based on eigenvalues
    ind=np.argsort(a[:,-1])
    a=a[ind]

    # trim zero-eigenvectors and eigenvalue col
    # a = a[3:,:-1]

    srtVal = a[:,-1:].flatten()
    srtVec = a[:,:-1]
    return(srtVal, srtVec)

def buildStiffnessMatrix(edges, nodes, noise_model, noise_stddev):
    """
    Stiffness matrix is actually FIM as derived in (J. Le Ny, ACC 2018)
    """
    numNodes = len(nodes)
    A = None
    K = np.zeros((numNodes*2, numNodes*2))
    alpha = None
    if noise_model == 'add':
        alpha = float(1)
    elif noise_model == 'lognorm':
        alpha = float(2)
    else:
        raise NotImplementedError

    for cnt, e in enumerate(edges):
        i, j = e
        if i == j:
            continue
        nodei = nodes[i]
        nodej = nodes[j]
        xi, yi = nodei.getXYLocation()
        xj, yj = nodej.getXYLocation()
        delXij = xi-xj
        delYij = yi-yj
        dist = np.sqrt(delXij**2 + delYij**2)

        #### If want to form matrix as A.T @ A
        # row = np.zeros(numNodes*2)
        # row[2*i] = delXij
        # row[2*i+1] = delYij
        # row[2*j] = -delXij
        # row[2*j+1] = -delYij
        # row = row/((noise_stddev)*(dist**alpha))
        # if A is None:
        #     A = row
        # else:
        #     A = np.vstack([A, row])

        Kii = np.array([[delXij**2,         delXij*delYij   ],
                        [delXij*delYij,     delYij**2       ]]) / ((noise_stddev**2) * (dist)**(2*alpha))
        Kij = -Kii
        # Kii
        K[2*i:2*i+2, 2*i:2*i+2] += Kii
        # Kjj
        K[2*j:2*j+2, 2*j:2*j+2] += Kii
        # Kij
        K[2*i:2*i+2, 2*j:2*j+2] = Kij
        # Kji
        K[2*j:2*j+2, 2*i:2*i+2] = Kij

    #### Testing different ways of building matrix
    # test = (A.T @ A)-K
    # print("TESTING")
    # matprintBlock(test)
    # print()
    # print()
    # matprintBlock(K)
    # print()
    # print()
    # matprintBlock(A.T @ A)

    return (K)

def groundNodesInMatrix(A, n, nodes):
    l = list(nodes)
    l.sort()
    for i in nodes:
        l.append(i+n)
    B = np.delete(A, l, axis=1)
    return B

####### Matrix Calculus #######
def getGradientOfMatrixForEigenpair(K, eigpair, graph):
    """
    Returns the gradient of the eigenval corresponding to the eigvec and matrix
    K

    :param      K:               Given matrix
    :type       K:               np.array(2n, 2n)
    :param      eigvec:          The eigenvector
    :type       eigvec:          np.array(2n)
    :param      eigval:          corresponding eigenval
    :type       eigval:          float

    :returns:   Gradient of eigenval as function of inputs
    :rtype:     np.array(2n)

    :raises     AssertionError:  Require that given vals are an eigenpair
    """

    assertIsEigpair(K, eigpair)
    assert(isSquareMatrix(K))

    eigval, eigvec = eigpair
    nrow, ncol = K.shape
    grad = np.zeros(nrow)

    for index in range(nrow):
        A = dKdVar(K, index, graph)
        grad[index] = quadraticMultiplication(eigvec, A)

    grad = grad/la.norm(grad,2)
    return grad

def dKdVar(K, index, graph):
    """
    Takes partial derivative of K w.r.t. input (v) and returns partial deriv of
    matrix

    :param      K:    input Matrix to take partial deriv of
    :type       K:    np.array(2n, 2n)
    :param      i:    index of input variable (eg (i = 0, v=x0) or (i=1, v=y0))
    :type       i:    int
    """
    size = K.shape
    v = ''

    i = (int)(np.floor(index/2))
    if index % 2 == 0:
        v = 'x'
    else:
        v = 'y'

    A = np.zeros_like(K)
    xi, yi = graph.getNodePositionTuple(i)

    node_i_connections = graph.getNodeConnectionList(i)
    for j in node_i_connections:
        xj, yj = graph.getNodePositionTuple(j)

        dKii_di = np.zeros((2,2))
        dKjj_di = np.zeros((2,2))
        dKij_di = np.zeros((2,2))
        dKji_di = np.zeros((2,2))
        if v is 'x':
            dKii_di = np.array( [[2*(xi-xj),        yi-yj   ],
                                [yi-yj,             0       ]])
            dKij_di = np.array( [[2*(xj-xi),        yj-yi   ],
                                [yj-yi,             0       ]])
            dKjj_di = dKii_di
            dKji_di = dKij_di
        elif v is 'y':
            dKii_di = np.array( [[0,                xi-xj   ],
                                [xi-xj,             2*(yi-yj)]])
            dKij_di = np.array( [[0,                xj-xi   ],
                                [xj-xi,             2*(yj-yi)]])
            dKjj_di = dKii_di
            dKji_di = dKij_di
        else:
            raise AssertionError

        # Kii
        A[2*i:2*i+2, 2*i:2*i+2] += dKii_di
        # Kjj
        A[2*j:2*j+2, 2*j:2*j+2] += dKjj_di
        # Kij
        A[2*i:2*i+2, 2*j:2*j+2] += dKij_di
        # Kji
        A[2*j:2*j+2, 2*i:2*i+2] += dKji_di


    return A

####### Lin. Alg. Utils #######
def quadraticMultiplication(vec, mat):
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
    quadMult =  vec.T @ mat @ vec
    return np.real(quadMult)

####### Random Generator Utils #######
def genRandomVector(nDim, length):
    vec = np.random.uniform(low=-2, high=2, size=nDim)
    vec = vec/np.linalg.norm(vec,2)
    vec *= length
    return vec

def genRandomTuple(lb=0, ub=10, size=2):
    vec = np.random.uniform(low=lb, high=ub, size=size)
    t = tuple(vec)
    return t

def genRandomLocation(xlb, xub, ylb, yub):
    xval = np.random.uniform(low=xlb, high=xub)
    yval = np.random.uniform(low=ylb, high=yub)
    return (xval, yval)

def CalculateLocalizationError(gnd_truth, est_locs):
    if not (gnd_truth.shape == est_locs.shape):
        print("Ground Truth Locs", gnd_truth)
        print("Estimated Locs", est_locs)
        return None
        # assert(gnd_truth.shape == est_locs.shape)
    n_rows, n_cols = gnd_truth.shape

    errors = []
    diff = gnd_truth - est_locs
    for row in range(n_rows):
        errors.append(la.norm(diff[row]))
    return errors


