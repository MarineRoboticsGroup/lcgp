"""
This class handles the constraint sets, which are sets of points on the
roadmap of the PRM that meet specific criteria

author: Alan Papalia (@alanpapalia)
"""


from typing import List, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np
import copy

# pylint: disable=import-error
import math_utils
import plot

colors = ["b", "g", "r", "c", "m", "y"]


class ConstraintSets:
    def __init__(self, robots, env, roadmap):
        self._robots = robots
        self._env = env
        self._roadmap = roadmap
        # Global Sets
        self.connected_states = [[set()] for x in range(robots.get_num_robots())]
        self.rigid_states = [[set()] for x in range(robots.get_num_robots())]
        self.conflict_states = [[set()] for x in range(robots.get_num_robots())]
        self.reachable_states = [[set()] for x in range(robots.get_num_robots())]
        self.valid_states = [[set()] for x in range(robots.get_num_robots())]
        self.last_timesteps = [[] for x in range(robots.get_num_robots())]
        self.updated_state = [False for _ in range(robots.get_num_robots())]
        self.updated_state[0] = True  # first robot never needs to update constraints

        # initialize first robot valid set
        self._init_first_robot_valid_set()
        # self.construct_valid_sets(0, None)

    def get_all_neighbors_of_set(self, loc_set: Set[int]):
        """return set of all direct neighbors

        Args:
            loc_set (set[int]): set of roadmap location indices

        Returns:
            set[int]: all direct neighbors to all locations in loc_set
        """
        neighbors = set()
        for loc in loc_set:
            neighbors.update(self._roadmap.get_connections(loc))
        return neighbors

    ###### Updates & Undos #######
    def undo_global_sets_updates(self, cur_robot_id):
        assert (
            False
        ), "temporary assertion to make sure that constraint sets aren't being incorrectly cleared"
        self.updated_state[cur_robot_id] = False
        self.connected_states[cur_robot_id] = [set()]
        self.rigid_states[cur_robot_id] = [set()]

    def _get_reachable_state_propagation(
        self, robot_id: int, timestep: int
    ) -> Set[int]:
        """Returns the states reachable at the given timestep based on
        propagation from the reachable states at the previous timestep

        Args:
            robot_id (int): the robot to find reachable states for
            timestep (int): the timestep to find reachable states for

        Returns:
            Set[int]: the reachable states at this timestep
        """
        # need to handle edge case for t=0
        if timestep == 0:
            start_loc = self._roadmap.get_start_index(robot_id)
            return set([start_loc])

        reachable_at_last_timestep = self.get_reachable_states(robot_id, timestep - 1)
        neighbors_last_timestep = self.get_all_neighbors_of_set(
            reachable_at_last_timestep
        )
        reachable_this_timestep = reachable_at_last_timestep.union(
            neighbors_last_timestep
        )
        return reachable_this_timestep

    def _get_valid_state_propagation(self, robot_id: int, timestep: int):
        assert isinstance(robot_id, int)
        assert isinstance(timestep, int)

        reachable_this_timestep = self.get_reachable_states(robot_id, timestep)
        assert isinstance(reachable_this_timestep, set)

        valid_states = set()
        if robot_id < 3:
            # update valid locations for this timestep
            all_reachable_are_valid = True
            if all_reachable_are_valid:
                for reachable_loc_id in reachable_this_timestep:
                    valid_states.add(reachable_loc_id)
                return valid_states
            else:
                connected_states = self.get_connected_states(robot_id, timestep)
                valid_states = connected_states.intersection(reachable_this_timestep)
                for valid_loc_id in valid_states:
                    valid_states.add(valid_loc_id)
                return valid_states
        else:
            rigid_this_timestep = self.get_rigid_states(robot_id, timestep)

            # assert rigid_this_timestep.issubset(
            #     reachable_this_timestep
            # ), f"{rigid_this_timestep}\n{reachable_this_timestep}\n{reachable_this_timestep - rigid_this_timestep }"

            valid_this_timestep = rigid_this_timestep.intersection(
                reachable_this_timestep
            )
            for valid_loc_id in valid_this_timestep:
                valid_states.add(valid_loc_id)
            return valid_states

    def _get_rigid_state_propagation(self, robot_id: int, timestep: int, trajs):
        assert isinstance(robot_id, int)
        assert robot_id >= 3
        assert isinstance(timestep, int)
        assert timestep >= 0

        reachable_this_timestep = self.get_reachable_states(robot_id, timestep)
        assert isinstance(reachable_this_timestep, set)

        conn_states = self.get_connected_states(robot_id, timestep)
        assert isinstance(conn_states, set)

        # only consider states that are connected and reachable
        rigid_candidate_states = conn_states.intersection(reachable_this_timestep)

        rigid_states_this_robot = set()
        rigid_states_next_robot = set()
        for candidate_loc_id in rigid_candidate_states:

            # if already found this to be rigid continue
            # need to first handle edge case of timestep being past length of
            # the current rigid states because otherwise we are checking if the
            # location was rigid at the previous timestep
            if timestep < len(self.rigid_states[robot_id]) and self.is_rigid_state(
                robot_id, timestep, candidate_loc_id
            ):
                continue

            state_rigid_this_robot, state_rigid_next_robot = self.state_would_be_rigid(
                trajs, robot_id, timestep, candidate_loc_id
            )
            assert isinstance(state_rigid_this_robot, bool)
            assert isinstance(state_rigid_next_robot, bool)

            if state_rigid_this_robot:
                rigid_states_this_robot.add(candidate_loc_id)
                if state_rigid_next_robot:
                    rigid_states_next_robot.add(candidate_loc_id)

        return rigid_states_this_robot, rigid_states_next_robot

    def _init_first_robot_valid_set(self):
        update_robot_id = 0
        start_id = self._roadmap.get_start_index(update_robot_id)
        goal_id = self._roadmap.get_goal_index(update_robot_id)

        # initialize these with starting location
        self.reachable_states[update_robot_id] = [set([start_id])]
        self.valid_states[update_robot_id] = [set([start_id])]

        timestep = 0
        # while goal is not in most recent valid state keep iterating
        while goal_id not in self.valid_states[update_robot_id][-1]:

            # update reachable locations for this timestep
            reachable_this_timestep = self._get_reachable_state_propagation(
                update_robot_id, timestep
            )
            for reachable_loc_id in reachable_this_timestep:
                self.add_reachable_state(update_robot_id, timestep, reachable_loc_id)

            # update valid locations for this timestep
            valid_this_timestep = self._get_valid_state_propagation(
                update_robot_id, timestep
            )
            for valid_loc_id in valid_this_timestep:
                self.add_valid_state(update_robot_id, timestep, valid_loc_id)

            timestep += 1

    def update_constraint_sets(self, trajs, cur_robot_id) -> Tuple[bool, int]:

        # cur_robot_id is the robot with the most recent trajectory planned
        assert cur_robot_id >= 0
        assert isinstance(cur_robot_id, int)

        # this is the robot whose constraint
        update_robot_id = cur_robot_id + 1
        assert update_robot_id < self._robots.get_num_robots()

        # this is the amount of time we want to build the valid set past the end of the trajectory
        time_past_traj = 5
        has_found_goal = False

        start_id = self._roadmap.get_start_index(update_robot_id)
        goal_id = self._roadmap.get_goal_index(update_robot_id)

        # if len(self.rigid_states[update_robot_id]) == 0:
        #     self.rigid_states[update_robot_id] = [set()]
        self.reachable_states[update_robot_id] = [set([start_id])]
        self.valid_states[update_robot_id] = [set([start_id])]

        curr_traj = trajs[cur_robot_id]

        # just add to connected_states
        for timestep, loc_id in enumerate(curr_traj):

            # update connected states for this timestep
            loc = self._roadmap.get_loc(loc_id)
            neighbors = self._roadmap.get_neighbors_within_radius(
                loc, self._robots._sensing_radius
            )
            for neighbor_id in neighbors:
                self.add_connected_state(update_robot_id, timestep, neighbor_id)

        # update other sets
        for timestep in range(len(curr_traj) + time_past_traj):

            # update reachable locations for this timestep
            reachable_this_timestep = self._get_reachable_state_propagation(
                update_robot_id, timestep
            )
            for reachable_loc_id in reachable_this_timestep:
                self.add_reachable_state(update_robot_id, timestep, reachable_loc_id)

            # for non-anchor robots we need to check rigidity
            if update_robot_id >= 3:
                (
                    rigid_this_timestep,
                    rigid_this_timestep_next_robot,
                ) = self._get_rigid_state_propagation(update_robot_id, timestep, trajs)
                for rigid_loc_id in rigid_this_timestep:
                    self.add_rigid_state(update_robot_id, timestep, rigid_loc_id)

                if update_robot_id + 1 < self._robots.get_num_robots():
                    for rigid_loc_id in rigid_this_timestep_next_robot:
                        self.add_rigid_state(
                            update_robot_id + 1, timestep, rigid_loc_id
                        )

            # update valid locations for this timestep
            valid_this_timestep = self._get_valid_state_propagation(
                update_robot_id, timestep
            )
            for valid_loc_id in valid_this_timestep:
                self.add_valid_state(update_robot_id, timestep, valid_loc_id)

            # if found goal id maybe we can exit?
            if goal_id in valid_this_timestep:
                has_found_goal = True

        if has_found_goal:
            return False, None
        else:
            return True, len(curr_traj) + time_past_traj

    ###### Check Status #######
    def state_would_be_rigid(
        self, trajs, cur_robot_id, cur_timestep, node_id
    ) -> Tuple[bool, bool]:
        """Checks a given node to see if it would be rigid given all of the already existing trajectories

        Args:
            trajs ([type]): [description]
            cur_robot_id ([type]): [description]
            cur_timestep ([type]): [description]
            node_id ([type]): [description]

        Returns:
            bool: whether the state would be rigid if the given robot occupied it
            bool: heuristic where if 2*rigid for robot 'i' will be rigid for
                robot 'i+1'
        """
        assert (
            cur_robot_id > 2
        ), "Trying to check rigidity without having any variable locations"
        # if not connected cannot be rigid
        is_connected = self.is_connected_state(cur_robot_id, cur_timestep, node_id)
        if not is_connected:
            return False, False

        num_neighbors = 0
        cur_node_loc = self._roadmap.get_loc(node_id)
        loc_list = []
        # iterate over positions of all already planned trajectories
        for robot_id in range(cur_robot_id):
            # build list of x,y locations
            other_robot_loc_id = self.get_location_id_at_time(
                trajs, robot_id, cur_timestep
            )
            other_robot_loc = self._roadmap.get_loc(other_robot_loc_id)
            loc_list.append(other_robot_loc)

            # count how many robots are within sensing radius
            dx = cur_node_loc[0] - other_robot_loc[0]
            dy = cur_node_loc[1] - other_robot_loc[1]
            dist_to_other_robot = np.hypot(dx, dy)
            if dist_to_other_robot < self._robots.get_sensing_radius():
                num_neighbors += 1
            if dist_to_other_robot < 1e-2:
                return False, False

        loc_list.append(cur_node_loc)

        # must be within sensing radius of 2 robots to be rigid
        if num_neighbors < 2:
            return False, False

        # either it wasn't a ManhattanRoadmap or we do not know this value
        # so we will calculate the rigidity directly
        is_rigid, next_robot_is_rigid = self._robots.test_rigidity_from_loc_list(
            loc_list
        )
        assert isinstance(is_rigid, bool), f"{type(is_rigid)}"
        assert isinstance(next_robot_is_rigid, bool), f"{type(next_robot_is_rigid)}"
        return is_rigid, next_robot_is_rigid

    def is_occupied_state(self, trajs, cur_robot_id, timestep, loc_id):
        for robot_id in range(cur_robot_id):
            if self.get_location_id_at_time(trajs, robot_id, timestep) == loc_id:
                return True
        return False

    def is_connected_state(self, cur_robot_id, timestep, loc_id):
        connStates = self.get_connected_states(cur_robot_id, timestep)
        if connStates:
            return loc_id in connStates
        else:
            return False

    def is_rigid_state(self, cur_robot_id: int, timestep: int, loc_id: int) -> bool:
        """returns whether a given state is rigid

        Args:
            cur_robot_id (int): [description]
            timestep (int): [description]
            loc_id (int): [description]

        Returns:
            bool: if the state is rigid
        """
        rigid_set = self.get_rigid_states(cur_robot_id, timestep)

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
        return self._roadmap.get_loc(trajs[cur_robot_id][time])

    def get_connected_states(self, cur_robot_id, timestep):
        # find most recent time to avoid checking state past end of robot
        # trajectory
        assert self.connected_states[cur_robot_id]
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
    def add_connected_state(self, cur_robot_id, timestep, loc_id):

        while len(self.connected_states[cur_robot_id]) < timestep + 1:
            self.connected_states[cur_robot_id].append(set())

        # if not already known, add state as connected
        if loc_id not in self.connected_states[cur_robot_id][timestep]:
            self.connected_states[cur_robot_id][timestep].add(loc_id)

            # go ahead and add to all future robots too
            if cur_robot_id + 1 < self._robots.get_num_robots():
                self.add_connected_state(cur_robot_id + 1, timestep, loc_id)

    def add_rigid_state(self, cur_robot_id, timestep, loc_id):
        assert self.is_connected_state(cur_robot_id, timestep, loc_id)

        while len(self.rigid_states[cur_robot_id]) < timestep + 1:
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

    def clear_conflicts(self, cur_robot_id):
        self.conflict_states[cur_robot_id].clear()

    ###### Plotting #######
    def animate_connected_states(self, cur_robot_id, coordTrajs, goalLocs):
        # plot.plot_trajectories(self._coord_trajs, self._robots, self._env, self._goal_locs)
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
                self._robots.get_robot_graph(),
                self._env,
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
            plt.title("Rigid States: Robot %d timestep %d" % (cur_robot_id, timestep))
            self.plot_rigid_states(cur_robot_id, timestep)
            for i, traj in enumerate(coordTrajs):
                if traj == []:
                    continue
                time = min(timestep, trajLens[i] - 1)
                loc = traj[time]
                plt.scatter(loc[0], loc[1], color=colors[i % 6])
                plt.plot(*zip(*traj), color=colors[i % 6])

            plot.plot(
                self._robots.get_robot_graph(),
                self._env,
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
        goal_id = self._roadmap.get_goal_index(cur_robot_id)
        goalLoc = self._roadmap.get_loc(goal_id)
        maxTimestep = len(self.valid_states[cur_robot_id])
        plt.close()
        trajLen = [len(traj) for traj in trajs]
        for timestep in range(maxTimestep):
            plot.clear_plot()
            plt.title("Valid States: Robot %d timestep %d" % (cur_robot_id, timestep))
            self.plot_connected_states(cur_robot_id, timestep)
            self.plot_rigid_states(cur_robot_id, timestep)
            plt.scatter(goalLoc[0], goalLoc[1], color="k")
            self.plot_valid_states(cur_robot_id, timestep)
            for i, traj in enumerate(trajs):
                if traj == []:
                    continue
                time = min(timestep, trajLen[i] - 1)
                loc = traj[time]
                plt.scatter(loc[0], loc[1], color=colors[i % 6])
                plt.plot(*zip(*traj), color=colors[i % 6], label="_nolegend_")
            plt.legend(["Connected", "Rigid", "GOAL", "Valid"])
            self.plot_environment()
            plt.show(block=False)
            plt.pause(0.1)

    def plot_connected_states(self, cur_robot_id, timestep):
        loc_ids = self.get_connected_states(cur_robot_id, timestep)
        pts = []
        for loc_id in loc_ids:
            pts.append(self._roadmap.get_loc(loc_id))
        xLocs = [x[0] for x in pts]
        yLocs = [x[1] for x in pts]
        plt.scatter(xLocs, yLocs, color="g")

    def plot_rigid_states(self, cur_robot_id, timestep):
        loc_ids = self.get_rigid_states(cur_robot_id, timestep)
        print(f"Rigid States for {cur_robot_id} at time {timestep}")
        pts = []
        for loc_id in loc_ids:
            pts.append(self._roadmap.get_loc(loc_id))
        xLocs = [x[0] for x in pts]
        yLocs = [x[1] for x in pts]
        plt.scatter(xLocs, yLocs, color="y")

    def plot_reachable_states(self, cur_robot_id, timestep):
        loc_ids = self.get_reachable_states(cur_robot_id, timestep)
        pts = []
        for loc_id in loc_ids:
            pts.append(self._roadmap.get_loc(loc_id))
        xLocs = [x[0] for x in pts]
        yLocs = [x[1] for x in pts]
        plt.scatter(xLocs, yLocs, color="b")

    def plot_valid_states(self, cur_robot_id, timestep):
        loc_ids = self.get_valid_states(cur_robot_id, timestep)
        pts = []
        for loc_id in loc_ids:
            pts.append(self._roadmap.get_loc(loc_id))
        xLocs = [x[0] for x in pts]
        yLocs = [x[1] for x in pts]
        plt.scatter(xLocs, yLocs, color="c")

    def plot_location_id_list(self, loc_ids):
        pts = []
        for loc_id in loc_ids:
            pts.append(self._roadmap.get_loc(loc_id))
        xLocs = [x[0] for x in pts]
        yLocs = [x[1] for x in pts]
        plt.scatter(xLocs, yLocs)

    def plot_environment(self):
        plot.plot_obstacles(self._env)
        plot.set_x_lim(self._env.get_bounds()[0], self._env.get_bounds()[1])
        plot.set_y_lim(self._env.get_bounds()[2], self._env.get_bounds()[3])
