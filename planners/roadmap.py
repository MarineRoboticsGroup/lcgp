import os
import chaospy
import numpy as np
import math
import string
from typing import List, Tuple
from abc import abstractmethod
import itertools

import kdtree
import math_utils
import rigidity_library


class Roadmap:
    """This object represents the probabilistic roadmap.

    Important: N_SAMPLE is not the number of locations in self.sample_locs. It is len(self.sample_locs) - 2*num_robots. This offset is due to the addition of the start and goal locations
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
        self.robots = robots
        self.env = env
        self.start_loc_list = self.robots.get_position_list_tuples()
        self.goalLocs = goalLocs
        self.N_SAMPLE = N_SAMPLE
        self.N_KNN = N_KNN
        self.MAX_EDGE_LEN = MAX_EDGE_LEN
        self.ROADMAP_TYPE = ROADMAP_TYPE

        if not self.ROADMAP_TYPE:
            self.ROADMAP_TYPE = self.__class__.__name__
            file_id = (
                f"{self.env.setting}_{self.robots.startConfig}_"
                f"{self.N_SAMPLE}samples_{self.N_KNN}nn_{self.MAX_EDGE_LEN}"
                f"len_{self.robots.get_num_robots()}rob_{self.ROADMAP_TYPE}"
            )
            cwd = os.getcwd()
            self.roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
            self.sample_locs_filename = (
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
                print("Read from existing roadmap file: %s" %
                      self.roadmap_filename)
                self.roadmap = roadmap
            else:
                print("%s not found. Generating Roadmap" %
                      self.roadmap_filename)
                self.roadmap = self.generate_roadmap()
                self.write_roadmap()
                print("New roadmap written to file")

        else:
            print("%s not found. Generating Sample Locs" %
                  self.sample_locs_filename)
            self.sample_locs = np.array(self.generate_sample_locs())
            self.nodeKDTree = kdtree.KDTree(self.sample_locs)
            self.write_sample_locs()
            print("New sample locations written to file")
            self.roadmap = self.generate_roadmap()
            self.write_roadmap()
            print("New roadmap written to file")

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
            index, _ = self.nodeKDTree.search(
                np.array(curLoc).reshape(2, 1), k=self.N_KNN
            )
            inds = index[0]
            edge_id = []
            for ii in range(1, len(inds)):
                connectingLoc = self.sample_locs[inds[ii]]
                if self.is_valid_path(curLoc, connectingLoc):
                    edge_id.append(inds[ii])
                    if len(edge_id) >= self.N_KNN:
                        break
            roadmap.append(edge_id)
        return roadmap

    ###### Accessors #######
    def get_connections(self, loc_id):
        return self.roadmap[loc_id]

    def get_loc(self, loc_id):
        return self.sample_locs[loc_id]

    def robots_would_collide(self, loc_id_1, loc_id_2, robot_size):
        loc_1 = self.get_loc(loc_id_1)
        loc_2 = self.get_loc(loc_id_2)
        x_dif = loc_1[0]-loc_2[0]
        y_dif = loc_1[1]-loc_2[1]
        return((-robot_size < x_dif < robot_size) or (-robot_size < y_dif < robot_size))

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
        index = self.N_SAMPLE + cur_robot_id
        return index

    def get_goal_index(self, cur_robot_id):
        index = self.N_SAMPLE + self.robots.get_num_robots() + cur_robot_id
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
        if dist >= self.MAX_EDGE_LEN:
            return False
        return self.env.is_valid_path(curLoc, connectingLoc)

    def read_roadmap(self):
        if not os.path.exists(self.roadmap_filename):
            return False
        roadmap = []
        with open(self.roadmap_filename, "r") as filehandle:
            for line in filehandle:
                roads = list(map(int, line.split()))
                roadmap.append(roads)
        return roadmap

    def write_roadmap(self):
        with open(self.roadmap_filename, "w") as filehandle:
            for roads in self.roadmap:
                line = str(roads).translate(
                    str.maketrans("", "", string.punctuation))
                filehandle.write("%s\n" % line)

    def read_sample_locs(self):
        if not os.path.exists(self.sample_locs_filename):
            return None
        sample_locs = []
        with open(self.sample_locs_filename, "r") as filehandle:
            for line in filehandle:
                loc = list(map(float, line.split()))
                sample_locs.append(loc)
        return sample_locs

    def write_sample_locs(self):
        file_dir = os.path.dirname(self.sample_locs_filename)
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir, exist_ok=True)

        with open(self.sample_locs_filename, "w") as filehandle:
            for sample_loc in self.sample_locs:
                # line = str(sample_loc)[1:-1]
                line = str(sample_loc).translate(str.maketrans("", "", "[],"))
                filehandle.write("%s\n" % line)


class ManhattanRoadmap(Roadmap):
    """
    # TODO finish docstring
    docstring
    """

    def __init__(
        self,
        robots,
        env,
        goalLocs: List[Tuple],
        N_SAMPLE: int,
        N_KNN: int,
        MAX_EDGE_LEN: float,
        NUM_ROWS: int,
        NUM_COLS: int,
        GRID_SPACING: float,
    ):
        self.NUM_ROWS = NUM_ROWS
        self.NUM_COLS = NUM_COLS
        self.GRID_SPACING = GRID_SPACING
        self.ROADMAP_TYPE = self.__class__.__name__

        super().__init__(
            robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN, self.ROADMAP_TYPE
        )

        file_id = (
            f"{self.env.setting}_{self.robots.startConfig}_"
            f"{GRID_SPACING}spacing_"
            f"{self.N_KNN}nn_{self.robots.get_num_robots()}rob_"
            f"{self.ROADMAP_TYPE}"
        )
        cwd = os.getcwd()
        self.roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
        self.sample_locs_filename = f"{cwd}/cached_planning/sample_locs_{file_id}.txt"

        self.rigidity_library = rigidity_library.RigidityLibrary(
            dist_between_nodes=self.GRID_SPACING,
            sensing_radius=self.robots.get_sensing_radius(),
            noise_stddev=self.robots.get_noise_stddev(),
            noise_model=self.robots.get_noise_model(),
            num_rows=self.NUM_ROWS,
            num_cols=self.NUM_COLS,
            max_num_robots=self.robots.get_num_robots(),
            multiproc=True,
        )

        super().init_sample_locs_and_roadmap()

        print(f"Init ManhattanRoadmap. FileID: {self.roadmap_filename}")

    def generate_sample_locs(self):
        """Generates a list of gridded points within the free space of the
        environment and then adds the start locations and goal locations of the
        planner to the list. Then returns the list

        Returns:
            List[List]: A list of the sample locations. Each second list is n (x,y) pair
        """

        xlb, xub, ylb, yub = self.env.bounds
        sample_locs = []
        x_locs = np.arange(xlb, xub, step=self.GRID_SPACING)
        y_locs = np.arange(ylb, yub, step=self.GRID_SPACING)
        locs = list(itertools.product(x_locs, y_locs))
        # If not within obstacle add location
        for loc in locs:
            if self.env.is_free_space(loc):
                sample_locs.append(list(loc))

        # N_SAMPLE is number of locs excluding start and goal

        for loc in self.start_loc_list:
            if loc in sample_locs:
                sample_locs.remove(loc)
            sample_locs.append(list(loc))
        for loc in self.goalLocs:
            if list(loc) in sample_locs:
                sample_locs.remove(list(loc))
            sample_locs.append(list(loc))

        self.N_SAMPLE = len(sample_locs) - \
            len(self.start_loc_list) - len(self.goalLocs)
        return sample_locs

    def get_cached_rigidity(self, loc_list: List[List]) -> bool:
        locs = np.array(loc_list)
        # num_robots = len(locs)
        if (locs % self.GRID_SPACING != 0).any():
            return None

        min_x = min(locs[:, 0])
        min_y = min(locs[:, 1])
        locs[:, 0] -= min_x
        locs[:, 1] -= min_y
        locs = (locs / self.GRID_SPACING).astype(int)
        # n_rows = max(locs[:,0])
        # n_cols = max(locs[:,1])
        loc_indices = (locs / self.GRID_SPACING).astype(int)
        loc_indices = [tuple(loc_index) for loc_index in loc_indices]
        # print(f"loc_indices: {loc_indices}")
        # rigidity_value = self.rigidity_library.get_rigidity_value(loc_indices, n_rows, n_cols, num_robots)
        rigidity_value = self.rigidity_library.get_rigidity_value(loc_indices)
        return rigidity_value


class RandomRoadmap(Roadmap):
    """
    docstring
    """

    def __init__(self, robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN):
        self.ROADMAP_TYPE = self.__class__.__name__
        self.N_SAMPLE = N_SAMPLE

        super().__init__(robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN)
        file_id = (
            f"{self.env.setting}_{self.robots.startConfig}_"
            f"{self.N_SAMPLE}samples_{self.N_KNN}nn_{self.MAX_EDGE_LEN}"
            f"len_{self.robots.get_num_robots()}rob_{self.ROADMAP_TYPE}"
        )
        cwd = os.getcwd()
        self.roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
        self.sample_locs_filename = f"{cwd}/cached_planning/sample_locs_{file_id}.txt"

        super().init_sample_locs_and_roadmap()

        print(f"Init RandomRoadmap. FileID: {self.roadmap_filename}")

    def generate_sample_locs(self) -> List[List]:
        """Generates a list of predetermined number of deterministic but
        randomly sampled locations in free space according to the halton
        distribution in the environment and then adds the start locations and
        goal locations of the planner to the list. Then returns the list

        Returns: List[List]: A list of the sample locations. Each second list is
            n (x,y) pair
        """
        xlb, xub, ylb, yub = self.env.bounds
        sample_locs = []
        distribution = chaospy.J(
            chaospy.Uniform(xlb + 0.1, xub -
                            0.1), chaospy.Uniform(ylb + 0.1, yub - 0.1)
        )
        samples = distribution.sample(self.N_SAMPLE * 10, rule="halton")
        i = 0
        while len(sample_locs) < self.N_SAMPLE and i < len(samples[0]):
            newLoc = samples[:, i]
            i += 1
            # If not within obstacle
            if self.env.is_free_space(newLoc):
                sample_locs.append(list(newLoc))
        if len(sample_locs) < self.N_SAMPLE:
            print("Not able to fully build roadmap. Need more samples")
            raise NotImplementedError
        for loc in self.start_loc_list:
            sample_locs.append(list(loc))
        for loc in self.goalLocs:
            sample_locs.append(list(loc))
        return sample_locs
