import os
import numpy as np
import itertools

# pylint: disable=import-error
from planners.roadmap.roadmap import Roadmap
from typing import List, Tuple
import rigidity_library


class ManhattanRoadmap(Roadmap):
    """
    this class is to represent a roadmap that is gridded up like a Manhattan
    world. This was originally intended to be used
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
        self._NUM_ROWS = NUM_ROWS
        self._NUM_COLS = NUM_COLS
        self._GRID_SPACING = GRID_SPACING
        self._ROADMAP_TYPE = self.__class__.__name__

        super().__init__(
            robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN, self._ROADMAP_TYPE
        )

        file_id = (
            f"{self._env.setting}_{self._robots._start_config}_"
            f"{GRID_SPACING}spacing_"
            f"{self._N_KNN}nn_{self._robots.get_num_robots()}rob_"
            f"{self._ROADMAP_TYPE}"
        )
        cwd = os.getcwd()
        self._roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
        self._sample_locs_filename = f"{cwd}/cached_planning/sample_locs_{file_id}.txt"

        self.rigidity_library = rigidity_library.RigidityLibrary(
            dist_between_nodes=self._GRID_SPACING,
            sensing_radius=self._robots.get_sensing_radius(),
            noise_stddev=self._robots.get_noise_stddev(),
            noise_model=self._robots.get_noise_model(),
            num_rows=self._NUM_ROWS,
            num_cols=self._NUM_COLS,
            max_num_robots=self._robots.get_num_robots(),
            multiproc=True,
        )

        super().init_sample_locs_and_roadmap()

        print(f"Init ManhattanRoadmap. FileID: {self._roadmap_filename}")

    def generate_sample_locs(self):
        """Generates a list of gridded points within the free space of the
        environment and then adds the start locations and goal locations of the
        planner to the list. Then returns the list

        Returns:
            List[List]: A list of the sample locations. Each second list is n (x,y) pair
        """

        xlb, xub, ylb, yub = self._env._bounds
        sample_locs = []
        x_locs = np.arange(xlb, xub, step=self._GRID_SPACING)
        y_locs = np.arange(ylb, yub, step=self._GRID_SPACING)
        locs = list(itertools.product(x_locs, y_locs))
        # If not within obstacle add location
        for loc in locs:
            if self._env.is_free_space(loc):
                sample_locs.append(list(loc))

        # N_SAMPLE is number of locs excluding start and goal

        for loc in self._start_loc_list:
            if loc in sample_locs:
                sample_locs.remove(loc)
            sample_locs.append(list(loc))
        for loc in self._goal_locs:
            if list(loc) in sample_locs:
                sample_locs.remove(list(loc))
            sample_locs.append(list(loc))

        self._N_SAMPLE = len(sample_locs) - len(self._start_loc_list) - len(self._goal_locs)
        return sample_locs

    def get_cached_rigidity(self, loc_list: List[List]) -> bool:
        locs = np.array(loc_list)
        # num_robots = len(locs)
        if (locs % self._GRID_SPACING != 0).any():
            return None

        min_x = min(locs[:, 0])
        min_y = min(locs[:, 1])
        locs[:, 0] -= min_x
        locs[:, 1] -= min_y
        locs = (locs / self._GRID_SPACING).astype(int)
        # n_rows = max(locs[:,0])
        # n_cols = max(locs[:,1])
        loc_indices = (locs / self._GRID_SPACING).astype(int)
        loc_indices = [tuple(loc_index) for loc_index in loc_indices]
        # print(f"loc_indices: {loc_indices}")
        # rigidity_value = self.rigidity_library.get_rigidity_value(loc_indices, n_rows, n_cols, num_robots)
        rigidity_value = self.rigidity_library.get_rigidity_value(loc_indices)
        return rigidity_value
