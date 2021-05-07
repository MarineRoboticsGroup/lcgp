"""

Localizability-Constrained Deployment of Mobile Robotic Networks
with Noisy Range Measurements

author: Jerome Le Ny and Simon Chauvière
modified: Nicole Thumma

"""

import numpy as np
import graph
import math_utils

class PotentialField():
    def __init__(self, robots, env, goals, min_eigval=0.75):
        self._robots = robots
        self._env = env
        self._goal_locs = goals
        self._start_loc_list = self._robots.get_position_list_tuples()
        self.min_eigval = min_eigval

        self.num_robots = robots.get_num_robots()

    def planning(self):
        """Returns trajectory
        """
        # TODO: figure out discrepency between size of matrices and connections in graph (error from get_partial_deriv_of_matrix)
        # Matrix is 4x4, but it seems to expect 10x10 matrix
        # Generally finish implementation, then test
        graph = self._robots.get_robot_graph()
        fim = math_utils.build_fisher_matrix(graph.edges, graph.nodes, graph.noise_model, graph.noise_stddev)
        eigvals = []
        eigvecs = []
        for i in range(fim.shape[0]):
            eigpair = math_utils.get_nth_eigpair(fim, i)
            eigvals.append(eigpair[0])
            eigvecs.append(eigpair[1])

        # Get min eigenval pair
        i = eigvals.index(min(eigvals))
        print("Index:", i)
        print("Min pair:", eigvals[i], eigvecs[i])

        return math_utils.get_gradient_of_eigpair(fim, (eigvals[i], eigvecs[i]), graph)
