import numpy as np
import scipy as sp
from scipy.sparse import linalg as sp_sparse_la
from scipy import linalg as scla
from numpy import linalg as npla
import time

def test_scipy_base(scipy_mat):
    # SCIPY BASE
    start = time.time()
    for i in range(10000):
        eigval = scla.eigvalsh(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigvals Hermitian: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval, _ = scla.eigh(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigpair Hermitian: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval, _ = scla.eig(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigpair: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval = scla.eigvals(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigval: {end-start}")

def test_scipy_sparse(scipy_mat):
    # start = time.time()
    # for i in range(10000):
    #     eigval = sp_sparse_la.eigsh(scipy_mat)
    # end = time.time()
    # print(f"Scipy Sparse SVD: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval, _ = sp_sparse_la.eigsh(scipy_mat)
    end = time.time()
    print(f"Scipy Sparse Eigpair Hermitian: {end-start}")


    start = time.time()
    for i in range(10000):
        eigval, _ = sp_sparse_la.eigs(scipy_mat)
    end = time.time()
    print(f"Scipy Sparse Eigpair: {end-start}")

    # start = time.time()
    # for i in range(10000):
    #     eigval = sp_sparse_la.eigvals(scipy_mat)
    # end = time.time()
    # print(f"Scipy Sparse Eigval: {end-start}")

def test_numpy(numpy_mat):
    # numpy
    start = time.time()
    for i in range(10000):
        eigval = npla.eigvalsh(numpy_mat)
    end = time.time()
    print(f"Numpy Eigvals Hermitian: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval, _ = npla.eigh(numpy_mat)
    end = time.time()
    print(f"Numpy Eigpair Hermitian: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval, _ = npla.eig(numpy_mat)
    end = time.time()
    print(f"Numpy Eigpair: {end-start}")

    start = time.time()
    for i in range(10000):
        eigval = npla.eigvals(numpy_mat)
    end = time.time()
    print(f"Numpy Eigval: {end-start}")

if __name__ == "__main__":
    """This is a basic test class to determine the relative speed of different methods for computing the eigenvalues of a matrix. I acknowledge that this doesn't necessarily use the flashiest linear algebra libraries but for smaller matrices such as those seen in our problem it may not make a difference
    """

    matrix = [[ 44.8,  22.4,  -0. ,  -0. , -16. ,  -0. ,  -8. ,  -8. , -12.8, -6.4,  -8. ,  -8. ,   0. ,   0. ,   0. ,   0. ], [ 22.4,  35.2,  -0. , -16. ,  -0. ,  -0. ,  -8. ,  -8. ,  -6.4, -3.2,  -8. ,  -8. ,   0. ,   0. ,   0. ,   0. ], [ -0. ,  -0. ,  67.2,   3.2,  -8. ,   8. , -16. ,  -0. , -16. , -0. , -12.8,  -6.4, -14.4,  -4.8,   0. ,   0. ], [ -0. , -16. ,   3.2,  28.8,   8. ,  -8. ,  -0. ,  -0. ,  -0. , -0. ,  -6.4,  -3.2,  -4.8,  -1.6,   0. ,   0. ], [-16. ,  -0. ,  -8. ,   8. ,  43.2,  14.4,  -0. ,  -0. ,  -8. , -8. ,  -3.2,  -6.4,  -8. ,  -8. ,   0. ,   0. ], [ -0. ,  -0. ,   8. ,  -8. ,  14.4,  52.8,  -0. , -16. ,  -8. , -8. ,  -6.4, -12.8,  -8. ,  -8. ,   0. ,   0. ], [ -8. ,  -8. , -16. ,  -0. ,  -0. ,  -0. ,  68.8,  30.4, -16. , -0. ,  -8. ,  -8. , -12.8,  -6.4,  -8. ,  -8. ], [ -8. ,  -8. ,  -0. ,  -0. ,  -0. , -16. ,  30.4,  43.2,  -0. , -0. ,  -8. ,  -8. ,  -6.4,  -3.2,  -8. ,  -8. ], [-12.8,  -6.4, -16. ,  -0. ,  -8. ,  -8. , -16. ,  -0. ,  64. , 28.8,  -0. ,  -0. ,  -8. ,  -8. ,  -3.2,  -6.4], [ -6.4,  -3.2,  -0. ,  -0. ,  -8. ,  -8. ,  -0. ,  -0. ,  28.8, 48. ,  -0. , -16. ,  -8. ,  -8. ,  -6.4, -12.8], [ -8. ,  -8. , -12.8,  -6.4,  -3.2,  -6.4,  -8. ,  -8. ,  -0. , -0. ,  56. ,  36.8, -16. ,  -0. ,  -8. ,  -8. ], [ -8. ,  -8. ,  -6.4,  -3.2,  -6.4, -12.8,  -8. ,  -8. ,  -0. , -16. ,  36.8,  56. ,  -0. ,  -0. ,  -8. ,  -8. ], [  0. ,   0. , -14.4,  -4.8,  -8. ,  -8. , -12.8,  -6.4,  -8. , -8. , -16. ,  -0. ,  59.2,  27.2,  -0. ,  -0. ], [  0. ,   0. ,  -4.8,  -1.6,  -8. ,  -8. ,  -6.4,  -3.2,  -8. , -8. ,  -0. ,  -0. ,  27.2,  36.8,  -0. , -16. ], [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,  -8. ,  -8. ,  -3.2, -6.4,  -8. ,  -8. ,  -0. ,  -0. ,  19.2,  22.4], [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,  -8. ,  -8. ,  -6.4, -12.8,  -8. ,  -8. ,  -0. , -16. ,  22.4,  44.8]]

    scipy_base = sp.array(matrix)
    scipy_sparse = sp.sparse.bsr_matrix(matrix)
    numpy_matrix = np.array(matrix)

    test_scipy_base(scipy_base)
    test_scipy_sparse(scipy_sparse)
    test_numpy(numpy_matrix)


