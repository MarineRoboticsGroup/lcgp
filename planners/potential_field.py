"""

Localizability-Constrained Deployment of Mobile Robotic Networks
with Noisy Range Measurements

Nicole Thumma

"""

import numpy as np
import math

# pylint: disable=import-error
import graph
import swarm
import math_utils


class PotentialField:
    def __init__(
        self,
        robots,
        env,
        goals,
        target_dist_to_goal: float = .5,
        max_move_dist: float = 1.0,
        max_iter: int = 2000,
    ):
        self._robots = robots
        self._env = env
        self._goal_locs = goals
        self._start_loc_list = self._robots.get_position_list_tuples()

        self.num_robots = robots.get_num_robots()
        self.found_goal = [False for i in range(self.num_robots)]
        self._trajs = [
            [self._start_loc_list[i]] for i in range(self.num_robots)
        ]  # list containing a list for each robot with tuples of postions

        # keep track of each robots distance to its goal
        self._curr_dist_to_goal = [
            self.getDistanceToGoal(idx) for idx in range(self.num_robots)
        ]
        self._current_robots = self._robots

        self._target_dist_to_goal = target_dist_to_goal
        self._max_move_dist = max_move_dist
        self._max_iter = max_iter

    def planning(self):
        """Returns trajectory
        """
        #DEBUGGING HELPERS
        # print("Start Positions:", [
        #     [round(self._trajs[i][-1][0], 3), round(self._trajs[i][-1][1], 3)] for i in range(self.num_robots)])
        recent_positions = []
        ##################

        #! relative weights aren't given in the paper, so these are guesses
        # normalize = False
        # w_task = 1000.0
        # w_loc = 100.0
        # w_avoid_obstacles = .001 #still causing trouble
        # w_avoid_robots = 0.0 #in line with how the others assume robot size of 0.0

        normalize = False
        w_task = 100.0
        w_loc = 10.0
        w_avoid_obstacles = .001 #still causing trouble
        w_avoid_robots = 0.0 #in line with how the others assume robot size of 0.0

        # normalize = True #probably not the right approach, but not confirmed, so it's still an option
        # w_task = 1
        # w_loc = 1
        # w_avoid_obstacles = 1
        # w_avoid_robots = 1


        # attempt at organizing the potential functions
        # weight, function, current value
        potentials = {
            "task": [w_task, self.f_task, []],
            "avoid_obstacles": [w_avoid_obstacles, self.f_avoid_obstacles, []],
            "avoid_robots": [w_avoid_robots, self.f_avoid_robots, []]
        }

        if self.num_robots > 3:
            potentials["loc"] = [w_loc, self.f_loc, []]

        for _ in range(self._max_iter):
            move_list = [[0.0, 0.0] for i in range(self.num_robots)]
            # potentials["task"][0] *= 1.1

            # get potential function values
            for name in potentials:
                potentials[name][2] = potentials[name][1](normalize)

            for robotIndex in range(self.num_robots):
                # Move towards goal
                if not self.hasFoundGoal(robotIndex):
                    # go to goal if within a step
                    if self.getDistanceToGoal(robotIndex) < self._max_move_dist:
                        #move_list[i] = goal - current pos
                        move_list[robotIndex][0] = self._goal_locs[robotIndex][0] - self._trajs[robotIndex][-1][0]
                        move_list[robotIndex][1] = self._goal_locs[robotIndex][1] - self._trajs[robotIndex][-1][1]

                    else:
                        # add potential functions together
                        for name in potentials:
                            print("\nCurrent robot:", robotIndex)
                            print("Current potential:", name)
                            # print("Current x value:", move_list[robotIndex][0])
                            #special case for anchors
                            if name == "loc" and robotIndex >= self.num_robots-3:
                                continue
                            move_list[robotIndex][0] += potentials[name][0] * \
                                potentials[name][2][2*robotIndex]
                            move_list[robotIndex][1] += potentials[name][0] * \
                                potentials[name][2][2*robotIndex+1]
                            # print("Print weighted addition:", potentials[name][0] * \
                            #     potentials[name][2][2*robotIndex])

                        # weight to move at max speed, might need to tune
                        hypot = math.hypot(
                            move_list[robotIndex][0], move_list[robotIndex][1]
                        )
                        move_list[robotIndex][0] *= self._max_move_dist / hypot
                        move_list[robotIndex][1] *= self._max_move_dist / hypot

                # stay at goal
                else:
                    move_list[robotIndex] = [0.0, 0.0]

                # Add movement to current position
                new_x = self._trajs[robotIndex][-1][0] + move_list[robotIndex][0]
                new_y = self._trajs[robotIndex][-1][1] + move_list[robotIndex][1]

                self._trajs[robotIndex].append(tuple([new_x, new_y]))

            # update for next timestep
            self._current_robots = swarm.Swarm(
                self._robots._sensing_radius,
                self._robots.noise_model,
                self._robots.noise_stddev,
            )
            self._current_robots.initialize_swarm_from_loc_list_of_tuples(
                [self._trajs[i][-1] for i in range(self.num_robots)]
            )

            # DEBUGGING HELPERS
            # if [[round(self._trajs[i][-1][0], 5), round(self._trajs[i][-1][1], 5)] for i in range(self.num_robots)] in recent_positions:
            #     print("STUCK IN A LOOP")
            #     print("Current positions:", [[round(self._trajs[i][-1][0], 5), round(self._trajs[i][-1][1], 5)] for i in range(self.num_robots)])
            #     print("Recent positions:", recent_positions)
            #     print("Potentials:", potentials)
            #     return

            recent_positions += [[[round(self._trajs[i][-1][0], 3), round(self._trajs[i][-1][1], 3)] for i in range(self.num_robots)]]
            if len(recent_positions) > 4:
                recent_positions = recent_positions[1:]

            # print("Current Positions:", [
            #     [round(self._trajs[i][-1][0], 3), round(self._trajs[i][-1][1], 3)] for i in range(self.num_robots)])
            ###################

        # End conditions
            if self.allRobotsFoundGoal():
                return self._trajs
        print("Not enough iterations")
        return self._trajs


####################
# Helper Functions #
####################

    def getDistanceToGoal(self, index):
        x, y = self._trajs[index][-1]
        xend, yend = self._goal_locs[index]
        dx = x - xend
        dy = y - yend
        return math.hypot(dx, dy)

    def hasFoundGoal(self, index):
        # if this is true then the robot will stop moving so we don't need to
        # check the distance in future timesteps
        if self._curr_dist_to_goal[index] < self._target_dist_to_goal:
            return True

        self._curr_dist_to_goal[index] = self.getDistanceToGoal(index)
        return self._curr_dist_to_goal[index] < self._target_dist_to_goal

    def allRobotsFoundGoal(self):
        for curr_dist in self._curr_dist_to_goal:
            if curr_dist > self._target_dist_to_goal:
                return False
        return True

    def detect_collision(self, index):
        # check distance from current robot to all other robots
        # check distance from current robot to all obstacles
        pass

#######################
# Potential Functions #
#######################

    # gradient of 1/2*sum_{i=1->n}(x_{i,current}-x_{i,goal})^2
    # => sum{i=1->n}((x_{i,current}-x_{i,goal}))
    def f_task(self, normalize):
        move_list = []
        for i in range(self.num_robots):
            # get unit vector towards goal
            x, y = self._trajs[i][-1]
            xend, yend = self._goal_locs[i]
            dx = xend - x
            dy = yend - y
            hypot = math.hypot(dx, dy)

            # if normalize:
            move_list += [dx / hypot, dy / hypot] #needed to avoid incentive decreasing as goal approaches
            # else:
            #     move_list += [dx, dy]
        return move_list

    def f_loc(self, normalize):
        # Get FIM
        graph = self._current_robots.get_robot_graph()
        fim = graph.get_fisher_matrix()

        # Get all eigval pairs
        eigvals = []
        eigvecs = []
        for i in range(fim.shape[0]):
            eigpair = math_utils.get_nth_eigpair(fim, i+1)
            eigvals.append(eigpair[0])
            eigvecs.append(eigpair[1])

        # Get min eigpair
        i = eigvals.index(min(eigvals))

        # Get gradient
        move_list = math_utils.get_gradient_of_eigpair(
            fim, (eigvals[i], eigvecs[i]), graph
        )

        # # normalize
        if normalize:
            for i in range(int(len(move_list) / 2)):
                hypot = math.hypot(move_list[i], move_list[i + 1])
                move_list[i] /= hypot
                move_list[i + 1] /= hypot
        return move_list

    def f_avoid_obstacles(self, normalize):
        robot_radius = 0.0 #! need to connect to actual parameter

        obstacle_list = self._env.get_obstacle_list()
        move_list = []
        for robotIndex in range(self.num_robots):
            x_r, y_r = self._trajs[robotIndex][-1]
            vec = [0.0, 0.0]
            for obstacle in obstacle_list:
                x_o, y_o = obstacle.get_center()
                obstacle_radius = obstacle.get_radius()
                dist = ((x_o - x_r)**2 + (y_o - y_r)**2)**1/2 - obstacle_radius - robot_radius
                if dist < 0.0001:
                    dist = 0.0001
                vec[0] -= (x_o - x_r)/dist**2
                vec[1] -= (y_o - y_r)/dist**2
            move_list += [vec[0]/len(obstacle_list), vec[1]/len(obstacle_list)] #avg vec

        # normalize
        if normalize:
            for i in range(int(len(move_list)/2)):
                hypot = math.hypot(move_list[i], move_list[i+1])
                move_list[i] /= hypot
                move_list[i+1] /= hypot
        return move_list

    def f_avoid_robots(self, normalize):
        robot_radius = 0.0 #! need to connect to actual parameter

        # get 1/(dist^2) between all robots
        # get weighted, normalized vector between all robots
        robot_vecs = [[[0.0, 0.0]
                       for j in range(self.num_robots)] for i in range(self.num_robots)]

        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                x_i, y_i = self._trajs[i][-1]
                x_j, y_j = self._trajs[j][-1]

                dist = ((x_j - x_i)**2 + (y_j - y_i)**2)**1/2 - 2*robot_radius
                if dist < 0:
                    dist = .0001
                vec = ((x_j - x_i)/dist**3, (y_j - y_i)/dist**3)

                robot_vecs[i][j] = (vec[0], vec[1])
                robot_vecs[j][i] = (-vec[0], -vec[1])

        move_list = []
        for i in range(self.num_robots):
            total_vec = [0.0, 0.0]

            # sum of weighted vectors
            for j in range(self.num_robots):
                total_vec[0] -= robot_vecs[i][j][0]
                total_vec[1] -= robot_vecs[i][j][1]

            # negate the sum
            move_list += [total_vec[0]/self.num_robots, total_vec[1]/self.num_robots] #avg vec

        # normalize
        if normalize:
            for i in range(int(len(move_list)/2)):
                hypot = math.hypot(move_list[i], move_list[i+1])
                move_list[i] /= hypot
                move_list[i+1] /= hypot
        return move_list