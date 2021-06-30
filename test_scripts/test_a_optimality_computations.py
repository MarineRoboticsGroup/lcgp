import numpy as np
import scipy as sp
from scipy import linalg as scla
from numpy import linalg as npla
import time

def test_scipy_base_eigvals(scipy_mat, num_iter):
    assert isinstance(num_iter, int)
    assert num_iter > 0
    # SCIPY BASE
    start = time.time()
    for _ in range(num_iter):
        eigval = scla.eigvalsh(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigvals Hermitian: {end-start}")

    start = time.time()
    for _ in range(num_iter):
        eigval, _ = scla.eigh(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigpair Hermitian: {end-start}")

    start = time.time()
    for _ in range(num_iter):
        eigval, _ = scla.eig(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigpair: {end-start}")

    start = time.time()
    for _ in range(num_iter):
        eigval = scla.eigvals(scipy_mat)
    end = time.time()
    print(f"Scipy Base Eigval: {end-start}")

def test_numpy_eigvals(numpy_mat, num_iter):
    assert isinstance(num_iter, int)
    assert num_iter > 0
    # numpy
    start = time.time()
    for _ in range(num_iter):
        eigval = npla.eigvalsh(numpy_mat)
    end = time.time()
    print(f"Numpy Eigvals Hermitian: {end-start}")

    start = time.time()
    for _ in range(num_iter):
        eigval, _ = npla.eigh(numpy_mat)
    end = time.time()
    print(f"Numpy Eigpair Hermitian: {end-start}")

    start = time.time()
    for _ in range(num_iter):
        eigval, _ = npla.eig(numpy_mat)
    end = time.time()
    print(f"Numpy Eigpair: {end-start}")

    start = time.time()
    for _ in range(num_iter):
        eigval = npla.eigvals(numpy_mat)
    end = time.time()
    print(f"Numpy Eigval: {end-start}")

def get_fim():

    matrix = [[ 44.8,  22.4,  -0. ,  -0. , -16. ,  -0. ,  -8. ,  -8. , -12.8, -6.4,  -8. ,  -8. ,   0. ,   0. ,   0. ,   0. ], [ 22.4,  35.2,  -0. , -16. ,  -0. ,  -0. ,  -8. ,  -8. ,  -6.4, -3.2,  -8. ,  -8. ,   0. ,   0. ,   0. ,   0. ], [ -0. ,  -0. ,  67.2,   3.2,  -8. ,   8. , -16. ,  -0. , -16. , -0. , -12.8,  -6.4, -14.4,  -4.8,   0. ,   0. ], [ -0. , -16. ,   3.2,  28.8,   8. ,  -8. ,  -0. ,  -0. ,  -0. , -0. ,  -6.4,  -3.2,  -4.8,  -1.6,   0. ,   0. ], [-16. ,  -0. ,  -8. ,   8. ,  43.2,  14.4,  -0. ,  -0. ,  -8. , -8. ,  -3.2,  -6.4,  -8. ,  -8. ,   0. ,   0. ], [ -0. ,  -0. ,   8. ,  -8. ,  14.4,  52.8,  -0. , -16. ,  -8. , -8. ,  -6.4, -12.8,  -8. ,  -8. ,   0. ,   0. ], [ -8. ,  -8. , -16. ,  -0. ,  -0. ,  -0. ,  68.8,  30.4, -16. , -0. ,  -8. ,  -8. , -12.8,  -6.4,  -8. ,  -8. ], [ -8. ,  -8. ,  -0. ,  -0. ,  -0. , -16. ,  30.4,  43.2,  -0. , -0. ,  -8. ,  -8. ,  -6.4,  -3.2,  -8. ,  -8. ], [-12.8,  -6.4, -16. ,  -0. ,  -8. ,  -8. , -16. ,  -0. ,  64. , 28.8,  -0. ,  -0. ,  -8. ,  -8. ,  -3.2,  -6.4], [ -6.4,  -3.2,  -0. ,  -0. ,  -8. ,  -8. ,  -0. ,  -0. ,  28.8, 48. ,  -0. , -16. ,  -8. ,  -8. ,  -6.4, -12.8], [ -8. ,  -8. , -12.8,  -6.4,  -3.2,  -6.4,  -8. ,  -8. ,  -0. , -0. ,  56. ,  36.8, -16. ,  -0. ,  -8. ,  -8. ], [ -8. ,  -8. ,  -6.4,  -3.2,  -6.4, -12.8,  -8. ,  -8. ,  -0. , -16. ,  36.8,  56. ,  -0. ,  -0. ,  -8. ,  -8. ], [  0. ,   0. , -14.4,  -4.8,  -8. ,  -8. , -12.8,  -6.4,  -8. , -8. , -16. ,  -0. ,  59.2,  27.2,  -0. ,  -0. ], [  0. ,   0. ,  -4.8,  -1.6,  -8. ,  -8. ,  -6.4,  -3.2,  -8. , -8. ,  -0. ,  -0. ,  27.2,  36.8,  -0. , -16. ], [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,  -8. ,  -8. ,  -3.2, -6.4,  -8. ,  -8. ,  -0. ,  -0. ,  19.2,  22.4], [  0. ,   0. ,   0. ,   0. ,   0. ,   0. ,  -8. ,  -8. ,  -6.4, -12.8,  -8. ,  -8. ,  -0. , -16. ,  22.4,  44.8]]
    return matrix

if __name__ == "__main__":
    """This is a basic test class to determine the relative speed of different
methods for computing the a-optimality from a FIM. I acknowledge that this
doesn't necessarily use the flashiest linear algebra libraries but for smaller
matrices such as those seen in our problem it may not make a difference
    """

    print("I did not finish implementing this script - needs further work")
    raise NotImplementedError

    matrix = get_fim()
    num_iter = 99999

    scipy_base = sp.array(matrix)
    numpy_matrix = np.array(matrix)

    test_scipy_base_eigvals(scipy_base, num_iter)
    test_numpy_eigvals(numpy_matrix, num_iter)


