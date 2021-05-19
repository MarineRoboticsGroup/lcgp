"""

Localizability-Constrained Deployment of Mobile Robotic Networks
with Noisy Range Measurements

Nicole Thumma

"""

import numpy as np
import graph
import math_utils
import math


class PotentialField():
    def __init__(self, robots, env, goals, min_eigval=0.75):
        self._robots = robots
        self._env = env
        self._goal_locs = goals
        self._start_loc_list = self._robots.get_position_list_tuples()
        self.min_eigval = min_eigval

        self.num_robots = robots.get_num_robots()
        self.found_goal = [False for i in range(self.num_robots)]
        self._trajs = [[self._start_loc_list[i]] for i in range(self.num_robots)] #need to doublecheck format

        self.goal_radius = .2
        self.max_iter = 20

    def planning(self):
        """Returns trajectory
        """
        # FIM working
        # TODO: Iterate through timesteps until goal reached by all agents
        # For each timestep:
        # get f_loc
        # get f_task
        # (get f_obstacle)
        # return weighted sum
        
        for i in range(self.max_iter):  
            for robotIndex in range(self.num_robots):
                if not self.hasFoundGoal(robotIndex):
                    return self.f_loc()

            if self.allRobotsFoundGoal():
                return self.getListOfFinalTrajectories()

        return None

    def getDistanceToGoal(self, index):
        x, y = self._trajs[index][-1]
        xend, yend = self._goal_locs[index]
        dx = x - xend
        dy = y - yend
        return math.hypot(dx, dy)

    def hasFoundGoal(self, index):
        if (self.getDistanceToGoal(index) < self.goal_radius):
            self.found_goal[robotIndex] = True
            return True
        return False

    def allRobotsFoundGoal(self):
        for res in self.found_goal:
            if not res:
                return False
        return True

    # gradient of 1/2*sum_{i=1->n}(x_{i,current}-x_{i,goal})^2
    def f_task(self):
        return None

    def f_loc(self):
        graph = self._robots.get_robot_graph()
        fim = math_utils.build_fisher_matrix(
            graph.edges, graph.nodes, graph.noise_model, graph.noise_stddev)
        eigvals = []
        eigvecs = []
        for i in range(fim.shape[0]):
            eigpair = math_utils.get_nth_eigpair(fim, i)
            eigvals.append(eigpair[0])
            eigvecs.append(eigpair[1])

        # Get min eigenval pair
        i = eigvals.index(min(eigvals))

        trajs = math_utils.get_gradient_of_eigpair(
            fim, (eigvals[i], eigvecs[i]), graph)
        print("Trajectories:", trajs)
        return trajs
