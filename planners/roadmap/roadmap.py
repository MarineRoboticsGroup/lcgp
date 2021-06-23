import os
import numpy as np
import math
import string
from typing import List, Tuple
from abc import abstractmethod

# pylint: disable=import-error
import kdtree
import math_utils


class Roadmap:
    """This object represents the probabilistic roadmap.

    Important: N_SAMPLE is not the number of locations in self.sample_locs.
    It is len(self.sample_locs) - 2*num_robots.  This offset is due to the
    addition of the start and goal locations
    """

    def __init__(
        self,
        robots,
        env,
        goalLocs,
        N_SAMPLE,
        N_KNN,
        MAX_EDGE_LEN,
        ROADMAP_TYPE: str = None,
    ):
        self._robots = robots
        self._env = env
        self._start_loc_list = self._robots.get_position_list_tuples()
        self._goal_locs = goalLocs
        self._N_SAMPLE = N_SAMPLE
        self._N_KNN = N_KNN
        self._MAX_EDGE_LEN = MAX_EDGE_LEN
        self._ROADMAP_TYPE = ROADMAP_TYPE

        if not self._ROADMAP_TYPE:
            self._ROADMAP_TYPE = self.__class__.__name__
            file_id = (
                f"{self._env.setting}_{self._robots._start_config}_"
                f"{self._N_SAMPLE}samples_{self._N_KNN}nn_{self._MAX_EDGE_LEN}"
                f"len_{self._robots.get_num_robots()}rob_{self._ROADMAP_TYPE}"
            )
            cwd = os.getcwd()
            self._roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
            self._sample_locs_filename = (
                f"{cwd}/cached_planning/sample_locs_{file_id}.txt"
            )

        # self.init_sample_locs_and_roadmap()

    def init_sample_locs_and_roadmap(self):
        """Initializes both the sample locs and the roadmap for the object. If there are sample_locs files already written somewhere, we will read those and try to read in corresponding roadmap files. If there are not sample_locs files written somewhere then we will generate sample_locs and a new roadmap and write both to a file."""

        # try to read sample locs from file. If doesn't exist then generate
        # new sample locs
        sample_locs = self.read_sample_locs()
        if sample_locs and len(sample_locs) > 0:
            self.sample_locs = sample_locs
            self.nodeKDTree = kdtree.KDTree(self.sample_locs)
            roadmap = self.read_roadmap()
            # try to read roadmap from file. If doesn't exist then generate new
            # roadmap
            if roadmap and (len(roadmap) > 0):
                print("Read from existing roadmap file: %s" % self._roadmap_filename)
                self._roadmap = roadmap
            else:
                print("%s not found. Generating Roadmap" % self._roadmap_filename)
                self._roadmap = self.generate_roadmap()
                self.write_roadmap()
                print("New roadmap written to file")

        else:
            print("%s not found. Generating Sample Locs" % self._sample_locs_filename)
            self.sample_locs = np.array(self.generate_sample_locs())
            self.nodeKDTree = kdtree.KDTree(self.sample_locs)
            # self.write_sample_locs()
            # print("New sample locations written to file")
            self._roadmap = self.generate_roadmap()
            # self.write_roadmap()
            # print("New roadmap written to file")

    @abstractmethod
    def generate_sample_locs(self) -> List[List]:
        pass

    def generate_roadmap(self):
        """
        Road map generation
        @return: list of list of edge ids ([[edges, from, 0], ...,[edges, from, N])
        """
        roadmap = []
        for curLoc in self.sample_locs:
            inds, _ = self.nodeKDTree.search(
                curLoc, k=self._N_KNN
            )
            edge_id = []
            for ii in range(1, len(inds)):
                connectingLoc = self.sample_locs[inds[ii]]
                if self.is_valid_path(curLoc, connectingLoc):
                    edge_id.append(inds[ii])
                    if len(edge_id) >= self._N_KNN:
                        break
            roadmap.append(edge_id)
        return roadmap

    ###### Accessors #######
    def get_connections(self, loc_id) -> List[int]:
        return self._roadmap[loc_id]

    def get_loc(self, loc_id) -> List[float]:
        return self.sample_locs[loc_id]

    def robots_would_collide(self, loc_id_1, loc_id_2, robot_size):
        loc_1 = self.get_loc(loc_id_1)
        loc_2 = self.get_loc(loc_id_2)
        x_dif = abs(loc_1[0] - loc_2[0])
        y_dif = abs(loc_1[1] - loc_2[1])

        return (x_dif < robot_size + 0.1 * robot_size) and (
            y_dif < robot_size + 0.1 * robot_size
        )

    def get_distance_between_loc_ids(self, loc_id_1, loc_id_2):
        loc_1 = self.get_loc(loc_id_1)
        loc_2 = self.get_loc(loc_id_2)
        dist = math_utils.calc_dist_between_locations(loc_1, loc_2)
        return dist

    def get_neighbors_within_radius(self, loc, radius):
        neighbor_indices = self.nodeKDTree.search_in_distance(loc, radius)
        return neighbor_indices

    def get_k_nearest_neighbors(self, loc, k):
        neighbor_indices, dists = self.nodeKDTree.search(loc, k)
        return neighbor_indices, dists

    def get_start_index(self, cur_robot_id):
        index = self._N_SAMPLE + cur_robot_id
        return index

    def get_goal_index(self, cur_robot_id):
        index = self._N_SAMPLE + self._robots.get_num_robots() + cur_robot_id
        return index

    ###### Conversions #######
    def convert_trajectories_to_coords(self, trajs):
        newTrajs = []
        for traj in trajs:
            newTraj = []
            for index in traj:
                newTraj.append(self.sample_locs[index])
                newTrajs.append(newTraj)
        return newTrajs

    def convert_trajectory_to_coords(self, traj):
        coords = []
        for index in traj:
            coords.append(tuple(self.sample_locs[index]))
        return coords

    ###### Utils #######
    def is_valid_path(self, curLoc, connectingLoc):
        dx = curLoc[0] - connectingLoc[0]
        dy = curLoc[1] - connectingLoc[1]
        dist = math.hypot(dx, dy)
        # node too far away
        if dist >= self._MAX_EDGE_LEN:
            return False
        return self._env.is_valid_path(curLoc, connectingLoc)

    def read_roadmap(self):
        if not os.path.exists(self._roadmap_filename):
            return False
        roadmap = []
        with open(self._roadmap_filename, "r") as filehandle:
            for line in filehandle:
                roads = list(map(int, line.split()))
                roadmap.append(roads)
        return roadmap

    def write_roadmap(self):
        with open(self._roadmap_filename, "w") as filehandle:
            for roads in self._roadmap:
                line = str(roads).translate(str.maketrans("", "", string.punctuation))
                filehandle.write("%s\n" % line)

    def read_sample_locs(self):
        if not os.path.exists(self._sample_locs_filename):
            return None
        sample_locs = []
        with open(self._sample_locs_filename, "r") as filehandle:
            for line in filehandle:
                loc = list(map(float, line.split()))
                sample_locs.append(loc)
        return sample_locs

    def write_sample_locs(self):
        file_dir = os.path.dirname(self._sample_locs_filename)
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir, exist_ok=True)

        with open(self._sample_locs_filename, "w") as filehandle:
            for sample_loc in self.sample_locs:
                # line = str(sample_loc)[1:-1]
                line = str(sample_loc).translate(str.maketrans("", "", "[],"))
                filehandle.write("%s\n" % line)
