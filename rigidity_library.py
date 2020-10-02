import math_utils
import graph

from typing import Tuple, List
import itertools
import tracemalloc
from sys import getsizeof
import numpy as np
import time
import os
import json


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
    ):
        assert noise_model in ["add", "lognorm",], (
            f"You passed in an invalid noise model for the"
            f"rigidity library: {noise_model}"
        )

        self.sensing_radius = sensing_radius
        self.noise_stddev = noise_stddev
        self.noise_model = noise_model
        self.dist_between_nodes = dist_between_nodes
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_num_robots = max_num_robots

        row_indices = range(self.num_rows)
        col_indices = range(self.num_cols)
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
            rows > self.num_rows
            or cols > self.num_cols
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

    def _generate_rigidity_dict(self, num_robots):
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
                x_pos = indices[0] * self.dist_between_nodes
                y_pos = indices[1] * self.dist_between_nodes
                pos = (x_pos, y_pos)
                node_locs.append(pos)
            return node_locs

        def write_locs_to_dict(combo, d):
            """Takes an iterable item containing possible locations on the checkerboard, computes corresponding rigidity values for these items, and then stores these values in a given dict

            Args:
                combo (iterable[Tuple]): iterable container of tuple of location
                    combinations on checkerboard
                d (dict): relates each tuple of locations to a given rigidity
                    value
            """
            test_graph = graph.Graph(self.noise_model, self.noise_stddev)
            for j, node_indices in enumerate(combo):

                # check if location is aligned on row 0 and col 0
                if not check_if_locs_aligned(node_indices):
                    continue

                # convert to XY style coordinates and then test the rigidity of
                # the corresponding graph
                node_locs = convert_indices_to_locs(node_indices)
                test_graph.initialize_from_location_list(node_locs, self.sensing_radius)
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

        print(
            f"Building Rigidity Library: {num_robots} robots, {self.dist_between_nodes} spacing, {self.num_rows} rows, {self.num_cols} cols"
        )
        rigidity_dict = {}
        possible_node_locs = itertools.combinations(
            self.all_possible_node_indices, num_robots
        )
        write_locs_to_dict(possible_node_locs, rigidity_dict)
        return rigidity_dict

    def init_rigidity_library(self):
        """
        Tries to read the rigidity library from a presaved file. If file not
        found, then we construct the library and then write it to file for
        future use
        """

        self.rigidity_library = self.read_rigidity_library()

        # if no rigidity library read, construct one
        if self.rigidity_library is None:
            self.rigidity_library = {}
            print(f"No rigidity library read from file. Constructing one now")
            for num_robot in range(3, self.max_num_robots + 1):
                self.rigidity_library[num_robot] = self._generate_rigidity_dict(
                    num_robot
                )

        cur_num_robots = self.get_current_num_robots()
        if cur_num_robots < self.max_num_robots:
            for num_robots in range(cur_num_robots, self.max_num_robots + 1):
                self.rigidity_library[num_robot] = self._generate_rigidity_dict(
                    num_robot
                )

        # write library to file so can be read in the future
        self.write_rigidity_library()
