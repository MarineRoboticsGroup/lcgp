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
import copy
from typing import List, Tuple, Set

# pylint: disable=import-error
from planners.roadmap import Roadmap, ManhattanRoadmap, RandomRoadmap
import math_utils
import plot
import swarm
import kdtree

colors = ["b", "g", "r", "c", "m", "y"]


class PriorityPrm:
    def __init__(self, robots, env, goals):
        # Roadmap Parameters
        self.N_SAMPLE = 850
        self.N_KNN = 10
        self.MAX_EDGE_LEN = 2
        self.NUM_ROWS = 15
        self.NUM_COLS = 15
        self.GRID_SPACING = 0.5
        # Swarm Parameters
        self.robots = robots
        self.sensingRadius = robots.get_sensing_radius()
        self.start_loc_list = self.robots.get_position_list_tuples()
        self.startConfig = self.robots.startConfig
        # environment
        self.env = env
        self.obstacles = env.get_obstacle_list()
        self.bounds = env.get_bounds()
        self.goalLocs = goals
        # Planning Constraints
        self.trajs = None
        self.coordTrajs = None
        # Necessary Objects
        manhattan_map = False
        if manhattan_map:
            self.roadmap = ManhattanRoadmap(
                self.robots,
                self.env,
                self.goalLocs,
                self.N_SAMPLE,
                self.N_KNN,
                self.MAX_EDGE_LEN,
                self.NUM_ROWS,
                self.NUM_COLS,
                self.GRID_SPACING,
            )
        else:
            self.roadmap = RandomRoadmap(
                self.robots,
                self.env,
                self.goalLocs,
                self.N_SAMPLE,
                self.N_KNN,
                self.MAX_EDGE_LEN,
            )

        # self.plot_roadmap()
        self.constraintSets = self.ConstraintSets(self.robots, self.env, self.roadmap)

    def planning(self, useTime=False):
        if not self.perform_planning(useTime):
            assert False, "The planning failed"
        print("Full set of trajectories found!")
        return self.coordTrajs

    def astar_planning(self, cur_robot_id, useTime):
        start_id = self.roadmap.get_start_index(cur_robot_id)
        goal_id = self.roadmap.get_goal_index(cur_robot_id)
        print("StartID", start_id, self.roadmap.get_loc(start_id))
        print("GoalID", goal_id, self.roadmap.get_loc(goal_id))
        startNode = self.Node(
            self.start_loc_list[cur_robot_id],
            cost=0.0,
            pind=-1,
            timestep=0,
            index=start_id,
            useTime=useTime,
        )
        goalNode = self.Node(
            self.goalLocs[cur_robot_id],
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
                conns = self.roadmap.get_connections(curNode.index)
                for i in range(len(conns)):
                    new_id = conns[i]
                    if new_id in closedSet:
                        continue
                    newLoc = self.roadmap.get_loc(new_id)
                    # dist = calc_dist_between_locations(curLoc, newLoc)
                    # newNode = self.Node(loc=newLoc, cost=curNode.cost + dist + 1, pind=curNode.index, timestep=curNode.timestep+1, index=new_id, useTime=useTime)
                    newNode = self.Node(
                        loc=newLoc,
                        cost=curNode.cost + 1,
                        pind=curNode.index,
                        timestep=curNode.timestep + 1,
                        index=new_id,
                        useTime=useTime,
                    )
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
            self.roadmap.get_goal_index(cur_robot_id)
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
        self.trajs = [[] for x in range(self.robots.get_num_robots())]
        self.coordTrajs = [[] for x in range(self.robots.get_num_robots())]
        cur_robot_id = 0
        while cur_robot_id < self.robots.get_num_robots():
            print("Planning for robot", cur_robot_id)
            timeStart = time.time()
            traj, success_planning = self.astar_planning(cur_robot_id, useTime)
            timeEnd = time.time()
            print(
                "Planning for robot %d completed in: %f (s)"
                % (cur_robot_id, timeEnd - timeStart)
            )
            # Was able to make traj
            if success_planning:
                self.trajs[cur_robot_id] = traj
                self.coordTrajs[
                    cur_robot_id
                ] = self.roadmap.convert_trajectory_to_coords(traj)
                # plot.plot_trajectories(self.coordTrajs, self.robots, self.env, self.goalLocs)
                # * Planning Success Condition
                if cur_robot_id == self.robots.get_num_robots() - 1:
                    print("All Planning Successful")
                    print()
                    return True

                # update based on new trajectory
                self.constraintSets.update_connected_sets_from_robot_traj(
                    self.trajs, cur_robot_id
                )
                hasConflict, conflictTime = self.constraintSets.construct_valid_sets(
                    cur_robot_id + 1, self.trajs
                )
                if hasConflict:
                    print(
                        "Found conflict planning for robot%d at time %d "
                        % (cur_robot_id, conflictTime)
                    )
                    self.constraintSets.animate_valid_states(
                        self.coordTrajs, cur_robot_id + 1
                    )
                    mintime = min(conflictTime, len(traj) - 1)
                    # conflictLocId = traj[conflictTime]
                    conflictLocId = traj[mintime]
                    print("Conflict at", conflictLocId, "at time", conflictTime, "\n")
                    self.constraintSets.undo_global_sets_updates(cur_robot_id)
                    self.constraintSets.add_conflict(
                        cur_robot_id, conflictTime, conflictLocId
                    )
                    self.reset_traj(cur_robot_id)
                else:
                    print("Planning successful for robot %d \n" % cur_robot_id)
                    if cur_robot_id == 0 and False:
                        self.constraintSets.animate_connected_states(
                            cur_robot_id + 1, self.coordTrajs, self.goalLocs
                        )
                    elif cur_robot_id > 0 and False:
                        self.constraintSets.animate_rigid_states(
                            cur_robot_id + 1, self.coordTrajs, self.goalLocs
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
        robot_size = 0.45
        for i in range(0, cur_robot_id):
            other_robot_loc_id = self.get_location_id_at_time(i, timestep)
            if self.roadmap.robots_would_collide(other_robot_loc_id, loc_id, robot_size):
                return False


        # * Trying heuristic tricks to get configuration to spread out more
        # if cur_robot_id == 1 :
        #     loc0 = self.get_location_at_time(0, timestep)
        # loc1 = self.roadmap.get_loc(loc_id)
        # if calc_dist_between_locations(loc0, loc1) < 1:
        #         return False
        # if cur_robot_id == 2:
        # loc0 = self.get_location_at_time(0, timestep)
        # loc1 = self.get_location_at_time(1, timestep)
        # loc2 = self.roadmap.get_loc(loc_id)
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
        self.trajs[cur_robot_id].clear()
        self.coordTrajs[cur_robot_id].clear()

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
        maxTime = len(self.trajs[cur_robot_id]) - 1
        time = min(maxTime, timestep)
        return self.trajs[cur_robot_id][time]

    def get_location_at_time(self, cur_robot_id, timestep):
        maxTime = len(self.trajs[cur_robot_id]) - 1
        time = min(maxTime, timestep)
        return self.roadmap.get_loc(self.trajs[cur_robot_id][time])

    ###### Conversion/Plotting/IO #######
    def plot_roadmap(self):
        print("Displaying Roadmap... May take time :)")
        edges = set()
        for i, _ in enumerate(self.roadmap.roadmap):
            for ii in range(len(self.roadmap.roadmap[i])):
                ind = self.roadmap.roadmap[i][ii]
                edge = (ind, i) if ind < i else (i, ind)
                if edge in edges:
                    continue
                else:
                    edges.add(edge)
                    plt.plot(
                        [
                            self.roadmap.sample_locs[i][0],
                            self.roadmap.sample_locs[ind][0],
                        ],
                        [
                            self.roadmap.sample_locs[i][1],
                            self.roadmap.sample_locs[ind][1],
                        ],
                        "-k",
                    )
        plot.plot_obstacles(self.env)
        plot.set_x_lim(self.env.get_bounds()[0], self.env.get_bounds()[1])
        plot.set_y_lim(self.env.get_bounds()[2], self.env.get_bounds()[3])

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
            path = [self.roadmap.get_loc(node.index), self.roadmap.get_loc(node.pind)]
            plt.plot(*zip(*path), color="b")
        plot.plot_obstacles(self.env)
        plot.plot_goals(self.goalLocs)
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

    class ConstraintSets:
        def __init__(self, robots, env, roadmap):
            self.robots = robots
            self.env = env
            self.roadmap = roadmap
            # Global Sets
            self.connected_states = [[set()] for x in range(robots.get_num_robots())]
            self.rigid_states = [[set()] for x in range(robots.get_num_robots())]
            self.conflict_states = [[set()] for x in range(robots.get_num_robots())]
            self.reachable_states = [[set()] for x in range(robots.get_num_robots())]
            self.valid_states = [[set()] for x in range(robots.get_num_robots())]
            self.last_timesteps = [[] for x in range(robots.get_num_robots())]

            # initialize first robot valid set
            self.construct_valid_sets(0, None)

        def construct_valid_sets(
            self, update_robot_id: int, trajs: List[List[Set[int]]]
        ) -> Tuple[bool, int]:
            """sets the reachable states and valid stat s for the s ecified
            robot. This requires having already updated the connected and rigid
            states for the given robot

            This function will return the time a conflict is found at if there
            is a conflict

            Args:
                update_robot_id (int): the robot to construct the valid sets
                    for
                trajs (list[list[set[int]]]): [description]

            Returns
                Tuple[bool, int]:
                    bool: true if has conflict, false if not
                    int: timestep of conflict, None if no conflict
            """
            # TODO make a testcase roadmap to check this
            # TODO add test to require that rigid and connected states were already updated
            assert update_robot_id >= 0
            start_id = self.roadmap.get_start_index(update_robot_id)
            goal_id = self.roadmap.get_goal_index(update_robot_id)
            reachable_sets_ = [set([start_id])]
            valid_sets_ = [set([start_id])]
            timestep = 0
            wait_for_valid_growth_cntr = 2
            time_past_goal_cntr = 2
            has_conflict = False
            while time_past_goal_cntr:

                # found goal location, add a few more sets just to be safe
                if goal_id in valid_sets_[timestep]:
                    time_past_goal_cntr -= 1

                # update reachable and valid sets
                for loc_id in reachable_sets_[timestep]:
                    self.add_reachable_state(update_robot_id, timestep, loc_id)

                for loc_id in valid_sets_[timestep]:
                    self.add_valid_state(update_robot_id, timestep, loc_id)

                # if valid set is empty return conflict where the conflict is
                # the time of the empty valid set
                if not valid_sets_[timestep]:
                    has_conflict = True
                    return has_conflict, timestep

                # if valid set hasn't added new locations for 2 counts then
                # return conflict at time when this began
                if not wait_for_valid_growth_cntr:
                    has_conflict = True
                    return has_conflict, timestep - 2

                # extend the size of the reachable and valid sets
                reachable_sets_.append(set())
                valid_sets_.append(set())

                neighbors = self.get_all_neighbors_of_set(valid_sets_[timestep])
                conflicts = self.get_conflict_states(update_robot_id, timestep + 1)

                if update_robot_id == 0:
                    occupied = set()
                else:
                    occupied = self.get_occupied_states(
                        trajs, update_robot_id, timestep + 1
                    )
                    # print(f"Occupied States for Robot {update_robot_id}: {occupied}")

                # reachable(t+1) = [valid(t) + neighbors of valid(t) -
                # conflicts(update_robot_id, t+1)] - occupied[t+1]
                reachable_sets_[timestep + 1] = (
                    valid_sets_[timestep]
                    .union(neighbors)
                    .difference(conflicts)
                    .difference(occupied)
                )

                if update_robot_id == 0:
                    valid_sets_[timestep + 1] = copy.deepcopy(
                        reachable_sets_[timestep + 1]
                    )
                elif update_robot_id == 1:
                    connected_states = self.get_connected_states(
                        update_robot_id, timestep + 1
                    )
                    valid_sets_[timestep + 1] = connected_states.intersection(
                        reachable_sets_[timestep + 1]
                    )
                else:
                    valid_sets_[timestep + 1] = self.get_rigid_subset(
                        trajs,
                        update_robot_id,
                        timestep + 1,
                        reachable_sets_[timestep + 1],
                    )

                timestep += 1

                # if valid set hasnt added new locations decrement counter
                if not valid_sets_[timestep].difference(valid_sets_[timestep - 1]):
                    wait_for_valid_growth_cntr -= 1
                elif wait_for_valid_growth_cntr < 2:
                    wait_for_valid_growth_cntr = 2

            self.reachable_states[update_robot_id] = reachable_sets_
            self.valid_states[update_robot_id] = valid_sets_

            # if successful return None as conflict
            return has_conflict, None

        def get_all_neighbors_of_set(self, loc_set: Set[int]):
            """return set of all direct neighbors

            Args:
                loc_set (set[int]): set of roadmap location indices

            Returns:
                set[int]: all direct neighbors to all locations in loc_set
            """
            neighbors = set()
            for loc in loc_set:
                neighbors.update(self.roadmap.get_connections(loc))
            return neighbors

        ###### Updates & Undos #######
        def undo_global_sets_updates(self, cur_robot_id):
            self.connected_states[cur_robot_id].clear()
            self.rigid_states[cur_robot_id].clear()

        def update_connected_sets_from_robot_traj(self, trajs, cur_robot_id):
            assert trajs[
                cur_robot_id
            ], "Tried to update based on nonexisting trajectory"

            update_robot_id = cur_robot_id + 1
            # copy over all current connected states because we know that won't
            # change. Do not copy over rigid though because that is nonmonotonic
            # property
            self.connected_states[update_robot_id] = copy.deepcopy(
                self.connected_states[cur_robot_id]
            )

            # add new states based on newest trajectory
            for timestep, loc_id in enumerate(trajs[cur_robot_id]):
                loc = self.roadmap.get_loc(loc_id)
                neighbors = self.roadmap.get_neighbors_within_radius(
                    loc, self.robots.sensingRadius
                )
                for location in neighbors:
                    self.add_connected_state(update_robot_id, timestep, location)
                self.connected_states[update_robot_id][timestep].update(neighbors)

        ###### Check Status #######
        # TODO review the use of this function
        def state_would_be_rigid(self, trajs, cur_robot_id, cur_timestep, node_id):
            """Checks a given node to see if it would be rigid given all of the already existing trajectories

            Args:
                trajs ([type]): [description]
                cur_robot_id ([type]): [description]
                cur_timestep ([type]): [description]
                node_id ([type]): [description]

            Returns:
                [type]: [description]
            """
            # if not connected cannot be rigid
            is_connected = self.is_connected_state(cur_robot_id, cur_timestep, node_id)
            if not is_connected:
                return False

            num_neighbors = 0
            cur_node_loc = self.roadmap.get_loc(node_id)
            loc_list = [cur_node_loc]
            # iterate over positions of all already planned trajectories
            for robot_id in range(cur_robot_id):
                # build list of x,y locations
                other_robot_loc_id = self.get_location_id_at_time(
                    trajs, robot_id, cur_timestep
                )
                other_robot_loc = self.roadmap.get_loc(other_robot_loc_id)
                loc_list.append(other_robot_loc)

                # count how many robots are within sensing radius
                dx = cur_node_loc[0] - other_robot_loc[0]
                dy = cur_node_loc[1] - other_robot_loc[1]
                dist_to_other_robot = np.hypot(dx, dy)
                if False:
                    print(f"Current Node Location: {cur_node_loc}")
                    print(f"Other Node Location: {other_robot_loc}")
                    print(f"Current Node ID: {node_id}")
                    print(f"Other Node ID: {other_robot_loc_id}")
                    print(f"Dist to Node {robot_id}: {dist_to_other_robot}")
                    print()
                if dist_to_other_robot < self.robots.get_sensing_radius():
                    num_neighbors += 1
                if dist_to_other_robot < 1e-2:
                    return False

            # must be within sensing radius of 2 robots to be rigid
            if num_neighbors < 2:
                return False

            # if using a manhattan roadmap we will check if it is a cached value
            # first
            if self.roadmap.ROADMAP_TYPE == "ManhattanRoadmap":
                rigidity = self.roadmap.get_cached_rigidity(loc_list)
                print(f"Loc List: {loc_list}, Cached Rigidity: {rigidity}")
                if rigidity is not None:
                    is_rigid = rigidity >= self.robots.min_eigval
                    return is_rigid

            # either it wasn't a ManhattanRoadmap or we do not know this value so we will calculate the rigidity directly
            is_rigid = self.robots.test_rigidity_from_loc_list(loc_list)
            return is_rigid

        def is_occupied_state(self, trajs, cur_robot_id, timestep, loc_id):
            for robot_id in range(cur_robot_id):
                if self.get_location_id_at_time(trajs, robot_id, timestep) == loc_id:
                    return True
            return False

        def is_connected_state(self, cur_robot_id, timestep, loc_id):
            connStates = self.get_connected_states(cur_robot_id, timestep)
            return loc_id in connStates

        def is_rigid_state(self, cur_robot_id, timestep, loc_id):
            rigid_set = self.get_rigid_states(cur_robot_id, timestep)
            # print("is_rigid_state:", loc_id, rigid_set)
            if rigid_set is None:
                return False
            return loc_id in rigid_set

        def is_valid_state(self, cur_robot_id, timestep, loc_id):
            valid_set = self.get_valid_states(cur_robot_id, timestep)
            return loc_id in valid_set

        def is_reachable_state(self, cur_robot_id, timestep, loc_id):
            reachSet = self.get_reachable_states(cur_robot_id, timestep)
            return loc_id in reachSet

        def is_conflict_state(self, cur_robot_id, timestep, loc_id):
            if timestep >= len(self.conflict_states[cur_robot_id]):
                return False
            isConflict = loc_id in self.conflict_states[cur_robot_id][timestep]
            #     print("Conflict found at time %d and location %d"%(timestep, loc_id))
            return isConflict

        ###### Getters #######
        def get_occupied_states(self, trajs, cur_robot_id, timestep):
            occupied_states = set()
            for robot_id in range(cur_robot_id):
                max_time = len(trajs[cur_robot_id]) - 1
                used_time = min(abs(max_time), timestep)
                cur_loc = trajs[robot_id][used_time]
                occupied_states.add(cur_loc)
            return occupied_states

        def get_location_id_at_time(self, trajs, cur_robot_id, timestep):
            maxTime = len(trajs[cur_robot_id]) - 1
            time = min(maxTime, timestep)
            return trajs[cur_robot_id][time]

        def get_location_at_time(self, trajs, cur_robot_id, timestep):
            maxTime = len(trajs[cur_robot_id]) - 1
            time = min(maxTime, timestep)
            return self.roadmap.get_loc(trajs[cur_robot_id][time])

        def get_connected_states(self, cur_robot_id, timestep):
            # find most recent time to avoid checking state past end of robot
            # trajectory
            maxTime = len(self.connected_states[cur_robot_id]) - 1
            time = min(maxTime, timestep)
            if time >= 0:
                conn_set = self.connected_states[cur_robot_id][time]
                if conn_set is None:
                    return set([])
                else:
                    return conn_set

        def get_rigid_states(self, cur_robot_id, timestep):
            # find most recent time to avoid checking state past end of robot
            # trajectory
            maxTime = len(self.rigid_states[cur_robot_id]) - 1
            time = min(maxTime, timestep)
            if time >= 0:
                rig_set = self.rigid_states[cur_robot_id][time]
                if rig_set is None:
                    return set([])
                else:
                    return rig_set

        def get_rigid_subset(self, trajs, cur_robot_id, timestep, original_set):
            rigid_subset = set(
                [
                    loc_id
                    for loc_id in original_set
                    if self.state_would_be_rigid(trajs, cur_robot_id, timestep, loc_id)
                ]
            )
            return rigid_subset

        def get_reachable_states(self, cur_robot_id, timestep):
            maxTime = len(self.reachable_states[cur_robot_id]) - 1
            time = min(maxTime, timestep)
            return self.reachable_states[cur_robot_id][time]

        def get_valid_states(self, cur_robot_id, timestep):
            maxTime = len(self.valid_states[cur_robot_id]) - 1
            time = min(maxTime, timestep)
            return self.valid_states[cur_robot_id][time]

        def get_conflict_states(self, cur_robot_id, timestep):
            max_time = len(self.conflict_states[cur_robot_id]) - 1
            if timestep > max_time:
                return set()
            return self.conflict_states[cur_robot_id][timestep]

        ###### Add/Remove/Clear #######
        # TODO make sure that connected states of all robot_id+1 is superset of robot_id
        # TODO something about setting the connected states is currently wrong
        def add_connected_state(self, cur_robot_id, timestep, loc_id):
            while len(self.connected_states[cur_robot_id]) <= timestep:
                self.connected_states[cur_robot_id].append(set())
            self.connected_states[cur_robot_id][timestep].add(loc_id)

        # TODO make sure that rigid states of all robot_id+1 is superset of robot_id
        def add_rigid_state(self, cur_robot_id, timestep, loc_id):
            assert self.is_connected_state(cur_robot_id, timestep, loc_id)
            while len(self.rigid_states[cur_robot_id]) <= timestep:
                self.rigid_states[cur_robot_id].append(set())
            self.rigid_states[cur_robot_id][timestep].add(loc_id)

        def add_reachable_state(self, cur_robot_id, timestep, loc_id):
            while len(self.reachable_states[cur_robot_id]) <= timestep:
                self.reachable_states[cur_robot_id].append(set())
            self.reachable_states[cur_robot_id][timestep].add(loc_id)

        def add_valid_state(self, cur_robot_id, timestep, loc_id):
            while len(self.valid_states[cur_robot_id]) <= timestep:
                self.valid_states[cur_robot_id].append(set())
            self.valid_states[cur_robot_id][timestep].add(loc_id)

        def add_conflict(self, cur_robot_id, timestep, loc_id):
            while len(self.conflict_states[cur_robot_id]) <= timestep:
                self.conflict_states[cur_robot_id].append(set())
            self.conflict_states[cur_robot_id][timestep].add(loc_id)

        def remove_connected_state(self, cur_robot_id, timestep, loc_id):
            assert self.is_connected_state(cur_robot_id, timestep, loc_id)
            self.connected_states[cur_robot_id][timestep].remove(loc_id)

        def remove_rigid_state(self, cur_robot_id, timestep, loc_id):
            assert self.is_rigid_state(cur_robot_id, timestep, loc_id)
            self.rigid_states[cur_robot_id][timestep].remove(loc_id)

        def clear_conflicts(self, cur_robot_id):
            self.conflict_states[cur_robot_id].clear()

        def clear_reachable_states(self, cur_robot_id):
            assert (
                False
            ), "This function shouldn't be used. It has been replaced by construct_valid_sets"
            self.reachable_states[cur_robot_id].clear()

        def clear_valid_states(self, cur_robot_id):
            assert (
                False
            ), "This function shouldn't be used. It has been replaced by construct_valid_sets"
            self.valid_states[cur_robot_id].clear()

        ###### Plotting #######
        def animate_connected_states(self, cur_robot_id, coordTrajs, goalLocs):
            # plot.plot_trajectories(self.coordTrajs, self.robots, self.env, self.goalLocs)
            print("Plotting Connected States")
            trajLens = [len(x) for x in coordTrajs]
            maxTimestep = max(trajLens)
            plt.close()
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title(
                    "Connected States: Robot %d timestep %d" % (cur_robot_id, timestep)
                )
                self.plot_connected_states(cur_robot_id, timestep)
                for i, traj in enumerate(coordTrajs):
                    time = min(timestep, trajLens[i] - 1)
                    if traj == []:
                        continue
                    loc = traj[time]
                    plt.scatter(loc[0], loc[1], color=colors[i % 6])
                    plt.plot(*zip(*traj), color=colors[i % 6])

                plot.plot(
                    self.robots.get_robot_graph(),
                    self.env,
                    blocking=False,
                    animation=True,
                    clear_last=False,
                    show_graph_edges=True,
                )
                # if timestep == 0:
                #     plt.pause(10)
            plt.close()

        def animate_rigid_states(self, cur_robot_id, coordTrajs, goalLocs):
            print(f"Plotting Rigid States for Robot {cur_robot_id}")
            trajLens = [len(x) for x in coordTrajs]
            maxTimestep = max(trajLens)
            plt.close()
            # plt.pause(5)
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title(
                    "Rigid States: Robot %d timestep %d" % (cur_robot_id, timestep)
                )
                self.plot_rigid_states(cur_robot_id, timestep)
                for i, traj in enumerate(coordTrajs):
                    if traj == []:
                        continue
                    time = min(timestep, trajLens[i] - 1)
                    loc = traj[time]
                    plt.scatter(loc[0], loc[1], color=colors[i % 6])
                    plt.plot(*zip(*traj), color=colors[i % 6])

                plot.plot(
                    self.robots.get_robot_graph(),
                    self.env,
                    blocking=False,
                    animation=True,
                    clear_last=False,
                    show_graph_edges=True,
                )
                # if timestep == 0:
                #     plt.pause(10)
            plt.close()

        def animate_reachable_states(self, cur_robot_id):
            print("Plotting Reachable States")
            maxTimestep = len(self.reachable_states[cur_robot_id])
            plt.close()
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title(
                    "Reachable States: Robot %d timestep %d" % (cur_robot_id, timestep)
                )
                self.plot_connected_states(cur_robot_id, timestep)
                self.plot_rigid_states(cur_robot_id, timestep)
                self.plot_reachable_states(cur_robot_id, timestep)
                plt.legend(["Connected", "Rigid", "Reachable"])
                self.plot_environment()
                plt.show(block=False)

        def animate_valid_states(self, trajs, cur_robot_id):
            print("Plotting Valid States")
            goal_id = self.roadmap.get_goal_index(cur_robot_id)
            goalLoc = self.roadmap.get_loc(goal_id)
            maxTimestep = len(self.valid_states[cur_robot_id])
            plt.close()
            # trajLen = [len(traj) for traj in trajs]
            for timestep in range(maxTimestep):
                plot.clear_plot()
                plt.title(
                    "Valid States: Robot %d timestep %d" % (cur_robot_id, timestep)
                )
                self.plot_connected_states(cur_robot_id, timestep)
                self.plot_rigid_states(cur_robot_id, timestep)
                self.plot_valid_states(cur_robot_id, timestep)
                plt.scatter(goalLoc[0], goalLoc[1], color="k")
                # for i, traj in enumerate(trajs):
                #     if traj == []:
                #         continue
                #     time = min(timestep, trajLen[i]-1)
                #     loc = traj[time]
                #     # loc = self.roadmap.get_loc(traj[time])
                #     plt.scatter(loc[0], loc[1], color=colors[i%6])
                #     plt.plot(*zip(*traj), color=colors[i%6])
                plt.legend(["Connected", "Rigid", "Valid", "GOAL"])
                self.plot_environment()
                plt.show(block=False)
                plt.pause(0.1)

        def plot_connected_states(self, cur_robot_id, timestep):
            loc_ids = self.get_connected_states(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.get_loc(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color="g")

        def plot_rigid_states(self, cur_robot_id, timestep):
            loc_ids = self.get_rigid_states(cur_robot_id, timestep)
            print(f"Rigid States for {cur_robot_id} at time {timestep}")
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.get_loc(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color="y")

        def plot_reachable_states(self, cur_robot_id, timestep):
            loc_ids = self.get_reachable_states(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.get_loc(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color="b")

        def plot_valid_states(self, cur_robot_id, timestep):
            loc_ids = self.get_valid_states(cur_robot_id, timestep)
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.get_loc(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs, color="c")

        def plot_location_id_list(self, loc_ids):
            pts = []
            for loc_id in loc_ids:
                pts.append(self.roadmap.get_loc(loc_id))
            xLocs = [x[0] for x in pts]
            yLocs = [x[1] for x in pts]
            plt.scatter(xLocs, yLocs)

        def plot_environment(self):
            plot.plot_obstacles(self.env)
            plot.set_x_lim(self.env.get_bounds()[0], self.env.get_bounds()[1])
            plot.set_y_lim(self.env.get_bounds()[2], self.env.get_bounds()[3])
