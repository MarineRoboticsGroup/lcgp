from sys import getsizeof
import scipy as sp
import scipy.sparse as sparse
import numpy as np
import time

def construct_random_arr(constructor, density:float, size:int):
    arr = np.zeros((n,n), dtype=bool)

    n_entries = size*size
    n_nonzero = int(n_entries*density)
    entry_locs = np.random.randint(0, size, size=(n_nonzero, 2))
    for i, j in entry_locs:
        arr[i, j] = 1
    new_arr = constructor(arr, dtype=bool)
    return new_arr


    for row in range(n):
        for col in range(n):
            arr[row, col] = np.random.rand() > p
    new_arr = constructor(arr, dtype=bool)
    return new_arr


def test_array_formation(constructor, density:float, size:int, num_reps:int)->float:
    start_time = time.time()
    for _ in range(num_reps):
        new_arr = construct_random_arr(constructor, density, size)
    end_time = time.time()
    return end_time-start_time

def test_array_hash(constructor, density:float, size:int, num_reps:int)->float:
    start_time = time.time()
    d = {}
    for _ in range(num_reps):
        new_arr = construct_random_arr(constructor, density, size)
        d[new_arr.getformat()] = 1
    end_time = time.time()
    return end_time-start_time

def test_array_sizes(constructor, density:float, size:int)->int:
    new_arr = construct_random_arr(constructor, density, size)
    return getsizeof(new_arr)

def test_dict_sizes(constructor, density:float, size:int, num_reps:int)->int:
    d = {}
    for _ in range(num_reps):
        new_arr = construct_random_arr(constructor, density, size)
        if constructor == np.array:
            d[new_arr.tostring()] = 1
        else:
            d[str(new_arr)] = 1
    # print(f"Number of entries: {len(d.keys())}")
    return getsizeof(d)



if __name__ == "__main__":
    """
        this is a set of test functions to determine what matrix representations
        are most time and space efficient for simple tasks such as matrix
        construction and hashing.

        Takeaways:
        - once hashed the size of the dict is the same regardless of the matrix
          used to generate hash.
        - All sparse matrix representations (except DOK) take up the same amount
          of memory and notably less than numpy
        - keeping array in numpy form is fastest way of constructing but if need
          to be a sparse representation use the COO form as it is much faster
          than others
        - speed of numpy construction (obviously) should scale near linearly
          with density
    """
    density = 0.05
    p = 1-density
    n = 100
    n_reps = 100

    # all of the different array constructors to test
    arr_types = [np.array, sparse.bsr_matrix, sparse.coo_matrix, sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix]

    for constructor in arr_types:
        construct_class = str(constructor)
        construct_class = construct_class[construct_class.find("\'")+1:]
        construct_class = construct_class[:construct_class.find("\'")]

        # construct_time = test_array_formation(constructor, density, n, n_reps)
        # print(f"constructed {n_reps} of {construct_class} in {construct_time} secs")

        # hash_time = test_array_formation(constructor, density, n, n_reps)
        # print(f"hashed {n_reps} of {construct_class} in {hash_time} secs")

        # arr_size = test_array_sizes(constructor, density, n)
        # print(f"{construct_class} size: {arr_size} bytes")

        # dict_size = test_dict_sizes(constructor, density, n, n_reps)
        # print(f"dict with {n_reps} {construct_class} entries: {dict_size} bytes")

        print()

