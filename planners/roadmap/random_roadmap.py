import os
import numpy as np
from typing import List, Tuple
import chaospy

# pylint: disable=import-error
from planners.roadmap.roadmap import Roadmap

class RandomRoadmap(Roadmap):
    """
    docstring
    """

    def __init__(self, robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN):
        self._ROADMAP_TYPE = self.__class__.__name__
        self._N_SAMPLE = N_SAMPLE

        super().__init__(robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN)
        file_id = (
            f"{self._env.setting}_{self._robots.start_config}_"
            f"{self._N_SAMPLE}samples_{self._N_KNN}nn_{self._MAX_EDGE_LEN}"
            f"len_{self._robots.get_num_robots()}rob_{self._ROADMAP_TYPE}"
        )
        cwd = os.getcwd()
        self._roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
        self._sample_locs_filename = f"{cwd}/cached_planning/sample_locs_{file_id}.txt"

        super().init_sample_locs_and_roadmap()

        print(f"Init RandomRoadmap. FileID: {self._roadmap_filename}")

    def generate_sample_locs(self) -> List[List]:
        """Generates a list of predetermined number of deterministic but
        randomly sampled locations in free space according to the halton
        distribution in the environment and then adds the start locations and
        goal locations of the planner to the list. Then returns the list

        Returns: List[List]: A list of the sample locations. Each second list is
            n (x,y) pair
        """
        xlb, xub, ylb, yub = self._env._bounds
        sample_locs = []
        distribution = chaospy.J(
            chaospy.Uniform(xlb + 0.1, xub - 0.1), chaospy.Uniform(ylb + 0.1, yub - 0.1)
        )
        samples = distribution.sample(self._N_SAMPLE * 10, rule="halton")
        i = 0
        while len(sample_locs) < self._N_SAMPLE and i < len(samples[0]):
            newLoc = samples[:, i]
            i += 1
            # If not within obstacle
            if self._env.is_free_space(newLoc):
                sample_locs.append(list(newLoc))
        if len(sample_locs) < self._N_SAMPLE:
            print("Not able to fully build roadmap. Need more samples")
            raise NotImplementedError
        for loc in self._start_loc_list:
            sample_locs.append(list(loc))
        for loc in self._goal_locs:
            sample_locs.append(list(loc))
        return sample_locs

