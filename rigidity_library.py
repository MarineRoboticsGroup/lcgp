import math_utils
import graph

import subprocess
import flamegraph
import multiprocessing
from typing import Tuple, List
import itertools
import tracemalloc
from sys import getsizeof
import numpy as np
import time
import os
import json


def _generate_rigidity_dict(
    rigidity_library,
    num_robots,
    dist_between_nodes,
    noise_model,
    noise_stddev,
    sensing_radius,
    num_rows,
    num_cols,
    all_possible_node_indices,
):
    def check_if_locs_aligned(node_indices) -> bool:
        """This function is to check if the nodes are properly aligned in
        the checkerboard frame, as many of the possible combinations will
        not be. This is because we intend to first align such that there is
        a robot on the 0th row and a robot on the 0th column of the
        checkerboard

        Args:
            node_indices (Tuple[Tuple]): Tuple of location tuples

        Returns:
            bool: whether or not the locations were aligned and thus valid
                to add to the dictionary
        """
        if not (0 == node_indices[0][0]):
            return False
        for loc in node_indices:
            if loc[1] == 0:
                return True
        return False

    def convert_indices_to_locs(node_indices):
        # Note: the x and y axes don't have to correspond with the
        # actual coordinates seen of a graph as long as there is
        # consistency in the labeling and the distances in x and y
        # directions are equal

        node_locs = []
        for indices in node_indices:
            x_pos = indices[0] * dist_between_nodes
            y_pos = indices[1] * dist_between_nodes
            pos = (x_pos, y_pos)
            node_locs.append(pos)
        return node_locs

    def transpose_is_in_set(added_locs, new_loc):
        def transpose(locs):
            loc_list = []
            for loc in locs:
                loc_list.append(tuple([loc[1], loc[0]]))
            sorted_list = sorted(loc_list, key=lambda element: (element[0], element[1]))
            locs = tuple(sorted_list)
            return locs

        trans_locs = transpose(new_loc)
        if trans_locs in added_locs:
            return True, trans_locs
        return False, None

    def write_locs_to_dict(combo, d):
        """Takes an iterable item containing possible locations on the checkerboard, computes corresponding rigidity values for these items, and then stores these values in a given dict

        Args:
            combo (iterable[Tuple]): iterable container of tuple of location
                combinations on checkerboard
            d (dict): relates each tuple of locations to a given rigidity
                value
        """
        test_graph = graph.Graph(noise_model, noise_stddev)
        locs_set = set(combo)
        locs_list = list(locs_set)
        added_locs = set()
        symmetry_counter = 0
        start_time = time.time()
        for j, node_indices in enumerate(locs_list):

            # check if location is aligned on row 0 and col 0
            if not check_if_locs_aligned(node_indices):
                locs_set.remove(node_indices)
                continue

            is_in_set, transpose_indices = transpose_is_in_set(added_locs, node_indices)
            if is_in_set:
                d[str(node_indices)] = d[str(transpose_indices)]
                symmetry_counter += 1
            else:
                added_locs.add(node_indices)
                # convert to XY style coordinates and then test the rigidity of
                # the corresponding graph
                node_locs = convert_indices_to_locs(node_indices)
                test_graph.initialize_from_location_list(node_locs, sensing_radius)
                rigidity_val = test_graph.get_nth_eigval(4)
                d[str(node_indices)] = rigidity_val

            if j % 2000 == 0 and False:
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

                assert (
                    current * 1e-9 < 3
                ), f"Overused memory. Stopping process. Current memory usage {current*1e-9} GB"

        end_time = time.time()
        print(
            f"Used symmetry {symmetry_counter} times for length {len(locs_list)} in {end_time-start_time} secs"
        )

    print(
        f"Building Rigidity Library: {num_robots} robots, {dist_between_nodes} spacing, {num_rows} rows, {num_cols} cols"
    )
    rigidity_dict = {}
    possible_node_locs = itertools.combinations(all_possible_node_indices, num_robots)
    write_locs_to_dict(possible_node_locs, rigidity_dict)
    rigidity_library[num_robots] = rigidity_dict


class RigidityLibrary:
    """This class is basically holding a bunch of dictionaries of precomputed
    rigidity checks. This is to attempt to leverage precomputation to reduce the
    amount of time spent checking the rigidity of a framework.

    Each library corresponds to a manhattan world style grid where every point
    in a graph is laid out in a checkerboard styling. Each library has a maximum
    number of robots it will account for.
    """

    def __init__(
        self,
        dist_between_nodes: float,
        sensing_radius: float,
        noise_stddev: float,
        noise_model: str,
        num_rows: int,
        num_cols: int,
        max_num_robots: int,
        multiproc: True,
    ):
        assert noise_model in ["add", "lognorm",], (
            f"You passed in an invalid noise model for the"
            f"rigidity library: {noise_model}"
        )

        self.sensing_radius = sensing_radius
        self.noise_stddev = noise_stddev
        self.noise_model = noise_model
        self.dist_between_nodes = dist_between_nodes
        self._NUM_ROWS = num_rows
        self._NUM_COLS = num_cols
        self.max_num_robots = max_num_robots
        self.multiproc = multiproc

        row_indices = range(self._NUM_ROWS)
        col_indices = range(self._NUM_COLS)
        self.all_possible_node_indices = list(
            itertools.product(row_indices, col_indices)
        )

        cwd = os.getcwd()
        self.filepath = (
            f"{cwd}/rigidity_dicts/"
            f"{num_rows}rows_{num_cols}cols_"
            f"{dist_between_nodes}spacing_"
            f"{noise_model}noise_{sensing_radius}radius.json"
        )
        self.init_rigidity_library()

    def increment_max_num_robots(self):
        assert (
            self.rigidity_library
        ), f"ERROR! Rigidity library is: {self.rigidity_library}"
        self.max_num_robots += 1
        self.rigidity_library[self.max_num_robots] = self._generate_rigidity_dict(
            self.max_num_robots
        )

    def get_current_num_robots(self):
        num_robots = max(self.rigidity_library)
        return int(num_robots)

    def has_cached_value(self, rows, cols, n_robots):
        if (
            rows > self._NUM_ROWS
            or cols > self._NUM_COLS
            or n_robots > self.max_num_robots
        ):
            return False
        else:
            return True

    def get_rigidity_value(self, locs: List[Tuple]):
        n_robots = len(locs)
        if n_robots > self.max_num_robots or n_robots < 3:
            return None
        locs = sorted(locs, key=lambda element: (element[0], element[1]))
        locs = tuple(locs)
        rigidity_value = self.rigidity_library.get(str(n_robots)).get(str(locs))
        return rigidity_value

    def read_rigidity_library(self):
        path_dir = os.path.dirname(self.filepath)
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)

        if not os.path.isfile(self.filepath):
            return None

        with open(self.filepath) as f:
            data = json.load(f)
        return data

    def write_rigidity_library(self):
        path_dir = os.path.dirname(self.filepath)
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        with open(self.filepath, "w") as f:
            json.dump(self.rigidity_library, f)

    def init_rigidity_library(self):
        """
        Tries to read the rigidity library from a presaved file. If file not
        found, then we construct the library and then write it to file for
        future use
        """

        # def _generate_rigidity_dict(num_robots, dist_between_nodes, noise_model, noise_stddev, sensing_radius, num_rows, num_cols, all_possible_node_indices):

        self.rigidity_library = self.read_rigidity_library()

        # if no rigidity library read, construct one
        if self.rigidity_library is None:
            print(f"No rigidity library read from file. Constructing one now")

            if self.multiproc:
                manager = multiprocessing.Manager()
                temp_rigidity_library = manager.dict()
                nproc = min(multiprocessing.cpu_count(), 14)
                pool = multiprocessing.Pool(nproc - 2)
            else:
                temp_rigidity_library = dict()

            for num_robots in range(3, self.max_num_robots + 1):
                if self.multiproc:
                    pool.apply_async(
                        _generate_rigidity_dict,
                        args=(
                            temp_rigidity_library,
                            num_robots,
                            self.dist_between_nodes,
                            self.noise_model,
                            self.noise_stddev,
                            self.sensing_radius,
                            self._NUM_ROWS,
                            self._NUM_COLS,
                            self.all_possible_node_indices,
                        ),
                    )
                else:
                    _generate_rigidity_dict(
                        temp_rigidity_library,
                        num_robots,
                        self.dist_between_nodes,
                        self.noise_model,
                        self.noise_stddev,
                        self.sensing_radius,
                        self._NUM_ROWS,
                        self._NUM_COLS,
                        self.all_possible_node_indices,
                    )

            if self.multiproc:
                pool.close()
                pool.join()

            self.rigidity_library = dict(temp_rigidity_library)
        else:
            cur_num_robots = self.get_current_num_robots()
            if cur_num_robots < self.max_num_robots:

                if self.multiproc:
                    manager = multiprocessing.Manager()
                    temp_rigidity_library = manager.dict()
                    nproc = min(multiprocessing.cpu_count(), 14)
                    pool = multiprocessing.Pool(nproc - 2)
                else:
                    temp_rigidity_library = dict()

                for num_robots in range(cur_num_robots, self.max_num_robots + 1):
                    if self.multiproc:
                        pool.apply_async(
                            _generate_rigidity_dict,
                            args=(
                                temp_rigidity_library,
                                num_robots,
                                self.dist_between_nodes,
                                self.noise_model,
                                self.noise_stddev,
                                self.sensing_radius,
                                self._NUM_ROWS,
                                self._NUM_COLS,
                                self.all_possible_node_indices,
                            ),
                        )
                    else:
                        _generate_rigidity_dict(
                            temp_rigidity_library,
                            num_robots,
                            self.dist_between_nodes,
                            self.noise_model,
                            self.noise_stddev,
                            self.sensing_radius,
                            self._NUM_ROWS,
                            self._NUM_COLS,
                            self.all_possible_node_indices,
                        )

                if self.multiproc:
                    pool.close()
                    pool.join()

                self.rigidity_library = dict(temp_rigidity_library)

        # write library to file so can be read in the future
        # self.write_rigidity_library()


if __name__ == "__main__":
    """Just a little testing of functionality"""
    dist_between_nodes = 1
    sensing_radius = 6
    noise_stddev = 0.65
    noise_model = "add"
    num_rows = 6
    num_cols = 6
    max_num_robots = 5

    profile = True
    if profile:
        cwd = os.getcwd()
        fg_log_path = f"{cwd}/profiling/test_rigidity_library.log"
        fg_thread = flamegraph.start_profile_thread(fd=open(fg_log_path, "w"))

    multiproc = False
    r = RigidityLibrary(
        dist_between_nodes,
        sensing_radius,
        noise_stddev,
        noise_model,
        num_rows,
        num_cols,
        max_num_robots,
        multiproc,
    )

    if profile:
        fg_thread.stop()
        fg_image_path = f"{cwd}/profiling/test_rigidity_library.svg"
        fg_script_path = f"{cwd}/flamegraph/flamegraph.pl"
        fg_bash_command = f"bash {cwd}/profiling/flamegraph.bash {fg_script_path} {fg_log_path} {fg_image_path}"
        subprocess.call(fg_bash_command.split(), stdout=subprocess.PIPE)
