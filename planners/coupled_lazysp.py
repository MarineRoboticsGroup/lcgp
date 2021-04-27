import numpy as np
from typing import List

# pylint: disable=import-error
from planners.roadmap.lazysp_roadmap import LazySpRoadmap, MetaState, MetaEdge
import math_utils
import plot
import swarm


class LazySp:
    def __init__(self, robots, env, goals):
        # Roadmap Parameters
        self._N_SAMPLE = 100
        self._N_KNN = 3
        self._MAX_EDGE_LEN = 2
        # Swarm Parameters
        self._robots = robots
        self._sensing_radius = self._robots.get_sensing_radius()
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
        # Roadmap
        self._roadmap = LazySpRoadmap(
            self._robots,
            self._env,
            self._goal_locs,
            self._N_SAMPLE,
            self._N_KNN,
            self._MAX_EDGE_LEN,
        )

        self._edges_evaluated = None

    def get_start_state(self) -> MetaState:
        loc_ids = []
        for robot_id in range(self._robots.get_num_robots()):
            robot_loc_id = self._roadmap.get_start_index(robot_id)
            loc_ids.append(robot_loc_id)

        start_state = MetaState(self._robots.get_num_robots(), loc_ids)
        return start_state

    def get_goal_state(self) -> MetaState:
        loc_ids = []
        for robot_id in range(self._robots.get_num_robots()):
            robot_loc_id = self._roadmap.get_goal_index(robot_id)
            loc_ids.append(robot_loc_id)

        goal_state = MetaState(self._robots.get_num_robots(), loc_ids)
        return goal_state

    def convert_state_path_to_edge_path(self, state_path):
        edge_path = []
        for i in range(len(state_path) - 1):
            edge = MetaEdge(state_path[i], state_path[i + 1], self._roadmap)
            edge_path.append(edge)
        return edge_path

    def perform_planning(self):
        """High level logic for LazySP (Dellin and Srinivasa 2016) to perform
        planning on a graph

        Returns:
            List[MetaEdges]: the path for the network represented as a
                list of edges in the joint space of the robots
        """
        print("Beginning planning for LazySP")
        # start with no edges evaluated
        self._edges_evaluated = set()

        # continuously loop over all paths we care about
        while True:

            # get shortest path with known edges
            candidate_state_path = self.get_shortest_path()

            candidate_edge_path = self.convert_state_path_to_edge_path(
                candidate_state_path
            )

            all_edges_evaluated = True

            # check to see if edges have been evaluated yet
            for edge in candidate_edge_path:
                if edge in self._edges_evaluated:
                    continue
                else:
                    all_edges_evaluated = False
                    break

            # if we found that all edges had been evaluated then we are done
            if all_edges_evaluated:
                return candidate_state_path

            # select edges to evaluate from the path
            selected_edges = self.select_edges(candidate_edge_path)
            unevaluated_selected_edges = selected_edges - self._edges_evaluated

            for edge in unevaluated_selected_edges:
                # add edge to list of evaluated edges
                self._edges_evaluated.add(edge)

                # check if edge passes rigidity constraint
                nonrigid_states = self.check_edge_states_for_rigidity(edge)
                if nonrigid_states is not None:
                    for state in nonrigid_states:
                        self._roadmap.set_nonrigid_state(state)
                    continue

                # check if edge contains a collision
                has_obstacle_collision = self.edge_has_collision(edge)
                if has_obstacle_collision:
                    self._roadmap.set_edge_cost(edge, np.inf)
                    continue

    def edge_has_collision(self, meta_edge: MetaEdge) -> bool:
        """checks all of the edges in the MetaEdge object to see if they run
        into an obstacle or go out of bounds in the environment

        Args:
            meta_edge (MetaEdge): the edge to check

        Returns:
            bool: True if MetaEdge had collision or went OOB, false otherwise
        """

        # get collection of all single robot edges
        edge_bundle = meta_edge.get_edge_bundle()

        # iterate over all edges
        for edge in edge_bundle:
            loc1_id, loc2_id = edge
            loc1 = self._roadmap.get_loc(loc1_id)
            loc2 = self._roadmap.get_loc(loc2_id)

            # check if runs into obstacle or goes out of bounds
            if not self._env.is_valid_path(loc1, loc2):
                return True

        # no edges were bad so return false
        return False

    # TODO test out other selectors
    def select_edges(self, path: List[MetaEdge]) -> List[MetaEdge]:
        return path

    def check_edge_states_for_rigidity(self, meta_edge: MetaEdge) -> List[MetaState]:
        state1, state2 = meta_edge.get_states()
        loc1_list = self._roadmap.get_state_locations(state1)
        loc2_list = self._roadmap.get_state_locations(state2)

        nonrigid_states = []
        if self._robots.test_rigidity_from_loc_list(loc1_list):
            nonrigid_states.append(state1)

        if self._robots.test_rigidity_from_loc_list(loc2_list):
            nonrigid_states.append(state2)

        return nonrigid_states

    def get_shortest_path(self) -> List[MetaState]:
        """This is the fundamental shortest path planner required for LazySP. We
        find the shortest path without ever actually evaluating the edge
        distances within this planner.

        Returns:
            List[MetaEdge]: [description]
        """
        start_state = self.get_start_state()
        goal_state = self.get_goal_state()

        dist_start_to_goal = self._roadmap.calc_dist_between_states(
            start_state, goal_state
        )
        start_node = self.LazySpNode(
            state=start_state,
            cost=0.0,
            prev_state=-1,
            heuristic_cost_to_goal=dist_start_to_goal,
        )
        goal_node = self.LazySpNode(
            state=goal_state, cost=0.0, prev_state=-1, heuristic_cost_to_goal=0.0
        )

        open_set, closed_set = dict(), dict()
        open_set[start_node] = start_node

        success = None
        while open_set:
            # if out of options, return conflict information
            if not open_set:
                success = False
                return ([], success)

            # find minimum cost in open_set. Note that the keys being used are
            # the states represented by the nodes
            cur_state = min(
                open_set,
                key=lambda o: open_set[o]._cost + open_set[o]._heuristic_cost_to_goal,
            )
            cur_node = open_set[cur_state]
            print(cur_node._heuristic_cost_to_goal)

            # Remove the item from the open set and add to closed set
            del open_set[cur_state]
            closed_set[cur_state] = cur_node

            # if goal location we can stop the search
            if cur_node.get_state() == goal_node.get_state():
                goal_node._prev_state = cur_node._prev_state
                goal_node._cost = cur_node._cost
                goal_node.timestep = cur_node.timestep
                break

            # get the possible states that this state could evolve from
            connected_metastates = self._roadmap.get_connecting_metastates(
                cur_node.get_state()
            )

            # iterate over connected states and add to list
            for new_state in connected_metastates:

                # if we have already closed this state we can move on
                if new_state in closed_set:
                    continue

                # build a new edge to get the edge cost estimate from the
                # roadmap. Assumption is there is minimal overhead in making
                # the MetaEdge object
                edge_cost = self._roadmap.calc_dist_between_states(
                    cur_node.get_state(), new_state
                )
                heur_cost_to_goal = self._roadmap.calc_dist_between_states(
                    new_state, goal_state
                )

                # make new node
                new_node = self.LazySpNode(
                    state=new_state,
                    cost=cur_node._cost + edge_cost,
                    prev_state=cur_node._prev_state,
                    heuristic_cost_to_goal=heur_cost_to_goal,
                )

                # Otherwise if it is already in the open set
                if new_state in open_set:
                    if open_set[new_state]._cost > new_node._cost:
                        open_set[new_state]._cost = new_node._cost
                        open_set[new_state]._prev_state = new_node._prev_state
                else:
                    open_set[new_state] = new_node

        # generate final path to return as candidate path
        # we build it from end to start and then reverse
        final_path = [goal_state]
        prev_state = goal_node._prev_state
        prev_state = goal_node._prev_state
        while prev_state is not None:
            final_path.append(prev_state)
            node = closed_set[prev_state]
            prev_state = node._prev_state
        final_path.reverse()

        success = True
        return (final_path, success)

    class LazySpNode:
        def __init__(
            self,
            state: MetaState,
            cost: float,
            prev_state: int,
            heuristic_cost_to_goal: float,
        ):
            self._state = state
            self._cost = cost
            self._prev_state = prev_state
            self._heuristic_cost_to_goal = heuristic_cost_to_goal

        def __str__(self):
            return (
                str(self._state) + ", " + str(self._cost) + ", " + str(self._prev_state)
            )

        def __hash__(self):
            return self._state.__hash__()

        def get_state(self):
            return self._state
