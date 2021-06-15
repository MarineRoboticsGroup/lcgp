"""
Probabilistic Road Map (PRM) Planner

author: Alan Papalia (@alanpapalia)
"""

import random
import math
import numpy as np
import chaospy
import scipy.spatial
import matplotlib.pyplot as plt
import os.path
from os import path
import os
import string
import time
from typing import List, Tuple, Set

# pylint: disable=import-error
from planners.roadmap.roadmap import Roadmap
from planners.roadmap.manhattan_roadmap import ManhattanRoadmap
from planners.roadmap.random_roadmap import RandomRoadmap
from planners.prioritized_planning.constraint_sets import ConstraintSets
import math_utils
import plot
import swarm
import kdtree


class PriorityPrm:
    def __init__(self, robots, env, goals):
        # Roadmap Parameters
        self._N_SAMPLE = 4*850
        self._N_KNN = 10
        self._MAX_EDGE_LEN = 1.0
        self._NUM_ROWS = 15
        self._NUM_COLS = 15
        self._GRID_SPACING = 0.25
        # Swarm Parameters
        self._robots = robots
        self._sensing_radius = robots.get_sensing_radius()
        self._start_loc_list = self._robots.get_position_list_tuples()
        self._start_config = self._robots._start_config
        # environment
        self._env = env
        self._obstacles = env.get_obstacle_list()
        self._bounds = env.get_bounds()
        self._goal_locs = goals
        # Planning Constraints
        self._trajs = None
        self._coord_trajs = None
        # Necessary Objects
        manhattan_map = False
        if manhattan_map:
            self._roadmap = ManhattanRoadmap(
                self._robots,
                self._env,
                self._goal_locs,
                self._N_SAMPLE,
                self._N_KNN,
                self._MAX_EDGE_LEN,
                self._NUM_ROWS,
                self._NUM_COLS,
                self._GRID_SPACING,
            )
        else:
            self._roadmap = RandomRoadmap(
                self._robots,
                self._env,
                self._goal_locs,
                self._N_SAMPLE,
                self._N_KNN,
                self._MAX_EDGE_LEN,
            )

        # self.plot_roadmap()
        self.constraintSets = ConstraintSets(self._robots, self._env, self._roadmap)

    def planning(self, useTime=False):
        if not self.perform_planning(useTime):
            assert False, "The planning failed"
        print("Full set of trajectories found!")
        return self._coord_trajs

    def astar_planning(self, cur_robot_id, useTime):
        start_id = self._roadmap.get_start_index(cur_robot_id)
        goal_id = self._roadmap.get_goal_index(cur_robot_id)
        print("StartID", start_id, self._roadmap.get_loc(start_id))
        print("GoalID", goal_id, self._roadmap.get_loc(goal_id))
        startNode = self.Node(
            self._start_loc_list[cur_robot_id],
            cost=0.0,
            pind=-1,
            timestep=0,
            index=start_id,
            useTime=useTime,
        )
        goalNode = self.Node(
            self._goal_locs[cur_robot_id],
            cost=0.0,
            pind=-1,
            timestep=-1,
            index=goal_id,
            useTime=useTime,
        )
        openSet, closedSet = dict(), dict()
        openSet[self.get_node_key(startNode, useTime)] = startNode
        success = False

        while True:
            # if out of options, return conflict information
            if not openSet:
                self.plot_failed_search(closedSet)
                return ([], success)

            # find minimum cost in openSet
            curKey = min(
                openSet,
                key=lambda o: openSet[o].cost
                + self.calc_heuristic(openSet[o], goalNode, useTime),
            )
            curNode = openSet[curKey]
            # Remove the item from the open set
            del openSet[curKey]
            closedSet[curKey] = curNode
            # If node is valid continue to expand path
            if self.node_is_valid(curNode, cur_robot_id):
                # if goal location
                if self.found_goal(curNode, goalNode):
                    goalNode.pind = curNode.pind
                    goalNode.cost = curNode.cost
                    goalNode.timestep = curNode.timestep
                    goalNode.pkey = curNode.pkey
                    break
                # Add it to the closed set
                closedSet[curKey] = curNode
                # curLoc = curNode.get_loc()
                # expand search grid based on motion model
                conns = self._roadmap.get_connections(curNode.index)
                for i in range(len(conns)):
                    new_id = conns[i]
                    if new_id in closedSet:
                        continue
                    newLoc = self._roadmap.get_loc(new_id)

                    dist = math_utils.calc_dist_between_locations(curNode.loc, newLoc)
                    newNode = self.Node(
                        loc=newLoc,
                        cost=curNode.cost + dist,
                        pind=curNode.index,
                        timestep=curNode.timestep + 1,
                        index=new_id,
                        useTime=useTime,
                    )

                    # newNode = self.Node(
                    #     loc=newLoc,
                    #     cost=curNode.cost + 1,
                    #     pind=curNode.index,
                    #     timestep=curNode.timestep + 1,
                    #     index=new_id,
                    #     useTime=useTime,
                    # )
                    newKey = self.get_node_key(newNode, useTime)
                    # Otherwise if it is already in the open set
                    if newKey in openSet:
                        if openSet[newKey].cost > newNode.cost:
                            # openSet[newKey] = newNode
                            openSet[newKey].cost = newNode.cost
                            openSet[newKey].pind = newNode.pind
                            openSet[newKey].timestep = newNode.timestep
                            openSet[newKey].pkey = newNode.pkey
                    else:
                        openSet[newKey] = newNode
        # generate final course
        path_indices = [
            self._roadmap.get_goal_index(cur_robot_id)
        ]  # Add goal node = goalNo
        pkey = goalNode.pkey
        pind = goalNode.pind
        while pind != -1:
            path_indices.append(pind)
            node = closedSet[pkey]
            pkey = node.pkey
            pind = node.pind
        path_indices.reverse()
        success = True
        return (path_indices, success)

    def perform_planning(self, useTime):
        print("Beginning Planning\n")
        self._trajs = [[] for x in range(self._robots.get_num_robots())]
        self._coord_trajs = [[] for x in range(self._robots.get_num_robots())]
        cur_robot_id = 0
        # while not done planning attempt to plan
        while cur_robot_id < self._robots.get_num_robots():
            print("Planning for robot", cur_robot_id)
            timeStart = time.time()
            traj, success_planning = self.astar_planning(cur_robot_id, useTime)
            timeEnd = time.time()
            print(
                "Planning for robot %d completed in: %f (s)"
                % (cur_robot_id, timeEnd - timeStart)
            )
            # * if successful in planning trajectory for this robot
            if success_planning:

                # * update trajectories
                self._trajs[cur_robot_id] = traj
                self._coord_trajs[
                    cur_robot_id
                ] = self._roadmap.convert_trajectory_to_coords(traj)
                # plot.plot_trajectories(self._coord_trajs, self._robots, self._env, self._goal_locs)

                # * if this was the last robot we can exit now
                if cur_robot_id == self._robots.get_num_robots() - 1:
                    print("All Planning Successful")
                    print()
                    return True
                else:

                    # * update connected and rigid sets for next robot
                    self.constraintSets.update_base_sets_from_robot_traj(
                        self._trajs, cur_robot_id
                    )

                    # * update valid sets for next robot now that we have
                    (
                        hasConflict,
                        conflictTime,
                    ) = self.constraintSets.construct_valid_sets(
                        cur_robot_id + 1, self._trajs
                    )


                    if hasConflict:
                        print(
                            "Found conflict planning for robot%d at time %d "
                            % (cur_robot_id, conflictTime)
                        )
                        # * animation
                        self.constraintSets.animate_valid_states(
                            self._coord_trajs, cur_robot_id + 1
                        )
                        mintime = min(conflictTime, len(traj) - 1)
                        # conflictLocId = traj[conflictTime]
                        conflictLocId = traj[mintime]
                        print(
                            "Conflict at", conflictLocId, "at time", conflictTime, "\n"
                        )
                        self.constraintSets.undo_global_sets_updates(cur_robot_id)
                        self.constraintSets.add_conflict(
                            cur_robot_id, conflictTime, conflictLocId
                        )
                        self.reset_traj(cur_robot_id)
                    else:
                        print("Planning successful for robot %d \n" % cur_robot_id)
                        # * animation
                        if False:
                            self.constraintSets.animate_valid_states(
                                self._coord_trajs, cur_robot_id + 1
                            )
                        self.constraintSets.clear_conflicts(cur_robot_id + 1)
                        cur_robot_id += 1
            else:
                print(
                    "Planning Failed for robot %d. \nReverting to plan for robot %d\n"
                    % (cur_robot_id, cur_robot_id - 1)
                )
                cur_robot_id -= 1

                if cur_robot_id < 0:
                    print("Failed to find paths")
                    return False

    ###### A* Helpers #######
    def node_is_valid(self, curNode, cur_robot_id):
        assert cur_robot_id >= 0
        loc_id = curNode.index
        timestep = curNode.timestep
        return self.state_is_valid(cur_robot_id, timestep, loc_id)

    # * this is where a lot of the magic happens!
    def state_is_valid(self, cur_robot_id, timestep, loc_id):
        assert cur_robot_id >= 0
        conflictFree = not self.constraintSets.is_conflict_state(
            cur_robot_id, timestep, loc_id
        )
        if not conflictFree:
            print(
                "Has Conflict, denying robot %d, time: %d, loc: %d"
                % (cur_robot_id, timestep, loc_id)
            )
            return False
        is_valid_state = self.constraintSets.is_valid_state(
            cur_robot_id, timestep, loc_id
        )
        if not is_valid_state:
            return False

        # check for collisions with other robots
        # will treat robot as a 0.4 m x 0.4 m square
        goal_id = self._roadmap.get_goal_index(cur_robot_id)

        robot_size = 0.4
        for other_robot_id in range(0, cur_robot_id):

            other_robot_loc_id = self.get_location_id_at_time(other_robot_id, timestep)
            if self._roadmap.robots_would_collide(
                other_robot_loc_id, loc_id, robot_size
            ):
                # print("Robots Collided")
                return False

            # iterate through all timestep/locations for robot i
            # because the current robot will sit at this location
            # after the planning is finished
            if loc_id == goal_id:
                # print("Checking goal location condition")
                max_time = len(self._trajs[other_robot_id]) - 1
                for _time in range(timestep, max_time + 1):
                    other_robot_loc_id = self.get_location_id_at_time(
                        other_robot_id, _time
                    )
                    if self._roadmap.robots_would_collide(
                        other_robot_loc_id, loc_id, robot_size
                    ):
                        # print(f"Robot {cur_robot_id} collided with robot {other_robot_id} due to goal location condition")
                        return False

        # * Trying heuristic tricks to get configuration to spread out more
        # if cur_robot_id == 1 :
        #     loc0 = self.get_location_at_time(0, timestep)
        # loc1 = self._roadmap.get_loc(loc_id)
        # if calc_dist_between_locations(loc0, loc1) < 1:
        #         return False
        # if cur_robot_id == 2:
        # loc0 = self.get_location_at_time(0, timestep)
        # loc1 = self.get_location_at_time(1, timestep)
        # loc2 = self._roadmap.get_loc(loc_id)
        # if calc_dist_between_locations(loc0, loc2) < 2:
        #     return False
        # if calc_dist_between_locations(loc1, loc2) < 1:
        #   return False
        return True

    def get_node_key(self, node, useTime):
        if useTime:
            return (node.index, node.timestep)
        else:
            return node.index

    def found_goal(self, curNode, goalNode):
        return curNode.index == goalNode.index

    def reset_traj(self, cur_robot_id):
        self._trajs[cur_robot_id].clear()
        self._coord_trajs[cur_robot_id].clear()

    ###### General Helpers #######
    def calc_heuristic(self, curNode, goalNode, useTime):
        curLoc = curNode.get_loc()
        goalLoc = goalNode.get_loc()
        nx, ny = curLoc
        gx, gy = goalLoc
        dx = gx - nx
        dy = gy - ny
        if useTime:
            return 0.5 * math.hypot(dx, dy)
        else:
            return math.hypot(dx, dy)

    def get_location_id_at_time(self, cur_robot_id, timestep):
        maxTime = len(self._trajs[cur_robot_id]) - 1
        time = min(maxTime, timestep)
        return self._trajs[cur_robot_id][time]

    def get_location_at_time(self, cur_robot_id, timestep):
        maxTime = len(self._trajs[cur_robot_id]) - 1
        time = min(maxTime, timestep)
        return self._roadmap.get_loc(self._trajs[cur_robot_id][time])

    ###### Conversion/Plotting/IO #######
    def plot_roadmap(self):
        print("Displaying Roadmap... May take time :)")
        edges = set()
        for i, _ in enumerate(self._roadmap._roadmap):
            for ii in range(len(self._roadmap._roadmap[i])):
                ind = self._roadmap._roadmap[i][ii]
                edge = (ind, i) if ind < i else (i, ind)
                if edge in edges:
                    continue
                else:
                    edges.add(edge)
                    plt.plot(
                        [
                            self._roadmap.sample_locs[i][0],
                            self._roadmap.sample_locs[ind][0],
                        ],
                        [
                            self._roadmap.sample_locs[i][1],
                            self._roadmap.sample_locs[ind][1],
                        ],
                        "-k",
                    )
        plot.plot_obstacles(self._env)
        plot.set_x_lim(self._env.get_bounds()[0], self._env.get_bounds()[1])
        plot.set_y_lim(self._env.get_bounds()[2], self._env.get_bounds()[3])

        plt.axis("off")
        plt.tick_params(
            axis="both",
            left="off",
            top="off",
            right="off",
            bottom="off",
            labelleft="off",
            labeltop="off",
            labelright="off",
            labelbottom="off",
        )
        plt.show(block=True)

    def plot_failed_search(self, closedSet):
        nodes = closedSet.values()
        print("Plotting the Failed Search!!")
        plot.clear_plot()
        plt.close()
        for node in nodes:
            if node.pind == -1:
                continue
            path = [self._roadmap.get_loc(node.index), self._roadmap.get_loc(node.pind)]
            plt.plot(*zip(*path), color="b")
        plot.plot_obstacles(self._env)
        plot.plot_goals(self._goal_locs)
        plt.show(block=True)
        # plot.showPlot()
        plt.close()

    ###### Member Classes #######
    class Node:
        def __init__(self, loc, cost, pind, timestep, index, useTime):
            self.loc = loc
            self.x = loc[0]
            self.y = loc[1]
            self.cost = cost
            self.pind = pind
            self.timestep = timestep
            self.index = index
            if useTime:
                self.pkey = (pind, timestep - 1)
            else:
                self.pkey = pind

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.pind)
            )

        def get_loc(self):
            return self.loc
