"""

Localizability-Constrained Deployment of Mobile Robotic Networks
with Noisy Range Measurements

Nicole Thumma

"""

import numpy as np
import graph
import swarm
import math_utils
import math


class PotentialField():
    def __init__(self, robots, env, goals, max_move_dist=0.2, max_iter=2000):
        self._robots = robots
        self._env = env
        self._goal_locs = goals
        self._start_loc_list = self._robots.get_position_list_tuples()

        self.num_robots = robots.get_num_robots()
        self.found_goal = [False for i in range(self.num_robots)]
        self._trajs = [[self._start_loc_list[i]]
                       for i in range(self.num_robots)]  # list containing a list for each robot with tuples of postions
        self._current_robots = self._robots

        self.max_move_dist = max_move_dist
        self.max_iter = max_iter

    def planning(self):
        """Returns trajectory
        """
        # print("Start Positions:", [
        #     [round(self._trajs[i][-1][0], 3), round(self._trajs[i][-1][1], 3)] for i in range(self.num_robots)])

        # relative weights aren't given in the paper, so these are guesses
        w_task = .75
        w_loc = .25

        for iteration in range(self.max_iter):
            move_list = [[0.0, 0.0] for i in range(self.num_robots)]
            if self.num_robots > 3:
                f_loc = self.f_loc()
            else:
                f_loc = []
                w_task = 1
                w_loc = 0
            f_task = self.f_task()

            for robotIndex in range(self.num_robots):
                # Move towards goal
                if not self.hasFoundGoal(robotIndex):
                    move_list[robotIndex][0] = w_task * f_task[2*robotIndex]
                    move_list[robotIndex][1] = w_task * f_task[2*robotIndex+1]

                    if robotIndex < self.num_robots-3:  # doublecheck which indices are anchors
                        move_list[robotIndex][0] += w_loc * f_loc[2*robotIndex]
                        move_list[robotIndex][1] += w_loc * \
                            f_loc[2*robotIndex+1]

                    # weight to move at max speed, might need to tune
                    hypot = math.hypot(
                        move_list[robotIndex][0], move_list[robotIndex][1])
                    move_list[robotIndex][0] *= self.max_move_dist/hypot
                    move_list[robotIndex][1] *= self.max_move_dist/hypot

                # stay at goal
                else:
                    move_list[robotIndex] = [0.0, 0.0]

                # Add movement to current position
                new_x = self._trajs[robotIndex][-1][0] + \
                    move_list[robotIndex][0]
                new_y = self._trajs[robotIndex][-1][1] + \
                    move_list[robotIndex][1]

                self._trajs[robotIndex].append(tuple([new_x, new_y]))

            # update for next timestep
            self._current_robots = swarm.Swarm(
                self._robots._sensing_radius, self._robots.noise_model, self._robots.noise_stddev)
            self._current_robots.initialize_swarm_from_loc_list_of_tuples(
                [self._trajs[i][-1] for i in range(self.num_robots)])

            # print("Current Positions:", [
            #     [round(self._trajs[i][-1][0], 3), round(self._trajs[i][-1][1], 3)] for i in range(self.num_robots)])

            if self.allRobotsFoundGoal():
                return self._trajs
        print("Not enough iterations")
        return None

    def getDistanceToGoal(self, index):
        x, y = self._trajs[index][-1]
        xend, yend = self._goal_locs[index]
        dx = x - xend
        dy = y - yend
        return math.hypot(dx, dy)

    def hasFoundGoal(self, index):
        if (self.getDistanceToGoal(index) < self.max_move_dist):
            self.found_goal[index] = True
            return True
        return False

    def allRobotsFoundGoal(self):
        for res in self.found_goal:
            if not res:
                return False
        return True

    # gradient of 1/2*sum_{i=1->n}(x_{i,current}-x_{i,goal})^2
    # => sum{i=1->n}((x_{i,current}-x_{i,goal}))
    def f_task(self):
        move_list = []
        for i in range(self.num_robots):
            # get unit vector towards goal
            x, y = self._trajs[i][-1]
            xend, yend = self._goal_locs[i]
            dx = xend - x
            dy = yend - y
            hypot = math.hypot(dx, dy)

            # normalize
            move_list += [dx/hypot, dy/hypot]
        return move_list

    def f_loc(self):
        # Get FIM
        graph = self._current_robots.get_robot_graph()
        fim = math_utils.build_fisher_matrix(
            graph.edges, graph.nodes, graph.noise_model, graph.noise_stddev)

        # Get all eigval pairs
        eigvals = []
        eigvecs = []
        for i in range(fim.shape[0]):
            eigpair = math_utils.get_nth_eigpair(fim, i)
            eigvals.append(eigpair[0])
            eigvecs.append(eigpair[1])

        # Get min eigenval pair
        i = eigvals.index(min(eigvals))

        # Get gradient
        move_list = math_utils.get_gradient_of_eigpair(
            fim, (eigvals[i], eigvecs[i]), graph)

        # normalize
        for i in range(int(len(move_list)/2)):
            hypot = math.hypot(move_list[i], move_list[i+1])
            move_list[i] /= hypot
            move_list[i+1] /= hypot
        return move_list
