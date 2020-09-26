import sys
from sys import getsizeof
from multiprocessing import Process, Manager
from multiprocessing import cpu_count, Queue, Pool
import multiprocessing

sys.path.insert(0, "./..")
import math_utils
import graph
import time
import itertools
import numpy as np
import logging

import queue
from guppy import hpy
import tracemalloc
import json


def count_iter(i):
    return sum(1 for e in i)


def write_iter_to_dict(combo, d):

    start_time = time.time()
    hp = hpy()
    for j, node_locs in enumerate(combo):

        if not (0 == node_locs[0][0]):
            continue

        aligned_col_0 = False
        for loc in node_locs:
            if loc[1] == 0:
                aligned_col_0 = True
                break

        if aligned_col_0:
            d[str(node_locs)] = 0

        # locs_arr = np.array(node_locs)
        # aligned_col_0 = (locs_arr[:,0] == 0).any()
        # aligned_row_0 = (locs_arr[:,1] == 0).any()
        # if not (aligned_col_0 and aligned_row_0):
            # continue


        if j % 2000 == 0 and False:
            print(hp.heap())
            print(f"Len Dict: {len(d)}")
            current, peak = tracemalloc.get_traced_memory()
            if True:
                print(
                    f"Current Memory: {current * 1e-9} GB"
                    + f"\nPeak Memory: {peak *1e-9} GB"
                    + f"\nDict Size : {getsizeof(d) * 1e-9} GB"
                    + f"\nNode Locs Size : {getsizeof(combo) * 1e-9} GB"
                    + "\n"
                )
            if current * 1e-9 > 3:
                print("WAYYY TOO MUCH MEMORY USAGE!!!")
                break
    end_time = time.time()
    print(f"Wrote iter to dict in {end_time-start_time} seconds")


def convert_itertools_to_form(combinations, form):
    time_start = time.time()
    res = form(combinations)
    time_end = time.time()
    print(f"Converted from itertools to {form} in {time_end-time_start} seconds")
    return res


def write_dict_to_json(filepath, d):
    with open(filepath, "w") as f:
        json.dump(d, f)


def generate_possible_indices(num_rows, num_cols):
    time_start = time.time()
    row_indices = np.arange(num_rows)
    col_indices = np.arange(num_cols)
    node_indices = itertools.product(row_indices, col_indices)
    time_end = time.time()
    print(f"Generated all possible indices in {time_end-time_start} seconds")
    return node_indices


def generate_possible_loc_combos(node_indices, num_robots):
    time_start = time.time()
    possible_node_locs = itertools.combinations(node_indices, num_robots)
    time_end = time.time()
    print(
        f"Generated all possible location combinations in {time_end-time_start} seconds"
    )
    return possible_node_locs


if __name__ == "__main__":
    """This is a script to test out the ability to precompute the rigidity of a
    bunch of different locations on a checkerboard style grid.

    Notions tested in this script are the speed of different ways of performing
    this precomputation, the memory requirement in caching different sizes, and
    results on multiprocessing implementations

    Results:

    Multiprocessing is more difficult to work with in this case than
    expected. It might be better to have the multiproc write everything to a set
    of files and then pull all related files together as needed. Assuming
    majority of computation will be in matrix methods anyways?

    """
    # Uncomment this line to see all multiprocessing output
    # multiprocessing.log_to_stderr(logging.DEBUG)

    multiproc = False

    # checkerboard parameters
    num_rows = 5
    num_cols = 5
    node_indices = list(generate_possible_indices(num_rows, num_cols))

    max_num_robots = 6
    cache_vals = []

    if multiproc:
        manager = Manager()
        nproc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=nproc * 2)

    for i, num_robots in enumerate(range(3, max_num_robots+1)):
        possible_node_locs = generate_possible_loc_combos(node_indices, num_robots)

        if multiproc:
            cache_vals.append(manager.dict())
            pool.apply_async(
                func=write_iter_to_dict, args=(possible_node_locs, cache_vals[i])
            )
        else:
            cache_vals.append({})
            write_iter_to_dict(possible_node_locs, cache_vals[i])

    if multiproc:
        pool.close()
        pool.join()

    for i, vals in enumerate(cache_vals):
        # print(vals)
        # print(len(vals))
        fp = f"./{i}_temp.json"
        write_dict_to_json(fp, vals)