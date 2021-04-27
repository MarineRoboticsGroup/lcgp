import os
import numpy as np
import math
import itertools
from typing import List, Tuple, Set
import chaospy

# pylint: disable=import-error
from planners.roadmap.roadmap import Roadmap


class MetaState:
    """This is a small class meant to represent the meta-state of our
    network of robots with a list of location ids
    """

    def __init__(self, num_robots: int, id_list: List[int]):
        assert num_robots == len(
            id_list
        ), "mismatch between the number of robots and number of location IDs provided"

        self._loc_ids = id_list
        self._id_string = str(id_list)
        self._num_robots = num_robots

    def get_loc_ids(self) -> List[int]:
        return self._loc_ids

    def get_num_robots(self) -> int:
        return self._num_robots

    def __str__(self):
        return self._id_string

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "_id_string") == self._id_string
        )

    def __hash__(self):
        return hash(self._id_string)

    def __gt__(self, other: "MetaState") -> bool:
        assert isinstance(other, self.__class__), "cannot compare different objects"
        return getattr(other, "_id_string") < self._id_string

    def __lt__(self, other: "MetaState") -> bool:
        assert isinstance(other, self.__class__), "cannot compare different objects"
        return getattr(other, "_id_string") > self._id_string

    def __ge__(self, other: "MetaState") -> bool:
        assert isinstance(other, self.__class__), "cannot compare different objects"
        return getattr(other, "_id_string") <= self._id_string

    def __le__(self, other: "MetaState") -> bool:
        assert isinstance(other, self.__class__), "cannot compare different objects"
        return getattr(other, "_id_string") >= self._id_string


class MetaEdge:
    """This class represents an edge between two MetaStates. This is
    primarily to keep track of which edges have been evaluated in the meta
    robot state
    """

    def __init__(self, state1, state2, roadmap):
        assert (
            state1.get_num_robots() == state2.get_num_robots()
        ), "tried to make edge between unbalanced number of states"

        self._states = [state1, state2]
        self._states.sort()
        self._edge_string = str(self._states[0]) + str(self._states[1])
        self._num_robots = state1.get_num_robots()
        self._edge_bundle = None

        # treat the distance as true distance between locs as defined in roadmap
        self._distance = roadmap.calc_dist_between_locations(state1, state2)

    def __str__(self):
        return self._edge_string

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "_edge_string") == self._edge_string
        )

    def __hash__(self):
        return hash(self._edge_string)

    def get_states(self):
        return self._states

    def get_edge_bundle(self) -> List[Tuple[int, int]]:
        """Returns a list of the edges w.r.t the single robots in the network.
        This can be thought of the list of edges that make up the 'MetaEdge'.
        This is analogous to how the list of the states of the robots make up
        the 'MetaState'.

        Returns:
            List[Tuple[int,int]]: the list of edges making up the MetaEdge,
                where edges are tuples of location ids
        """
        if self._edge_bundle is not None:
            return self._edge_bundle

        state1, state2 = self._states
        loc_id_1 = state1.get_loc_ids()
        loc_id_2 = state2.get_loc_ids()
        edge_bundle = []
        for i in range(self._num_robots):
            edge_pair = (loc_id_1[i], loc_id_2[i])
            edge_bundle.append(edge_pair)

        self._edge_bundle = edge_bundle

    def get_edge_cost(self) -> float:
        return self._distance


class LazySpRoadmap(Roadmap):
    """This is a roadmap to be used for high-dimensional planning via LazySP
    (Dellin and Srinivasa 2016).

    The idea is to build a roadmap without ever checking for any collisions
    (so edges allow for running through objects) and the planner will plan
    paths on this graph and only evaluate the edges which are along the paths
    - so only checking for collisions during this path evaluation phase.

    Because of the prohibitive cost of building a high-dimensional state lattice and
    actually representing all of the edges, we choose to take the path of
    building a 2-D roadmap and providing a 'get_edges' function to determine
    what the neighboring states are.

    As a first attempt at providing the functionality of removing edges we
    will keep track of the state-to-state transitions that are prevented.

    NOTE: the viewpoint on this roadmap is that the set of all robot locations
    at a given time is the "state", ie we are treating the network of robots as
    a single meta-agent and planning in the joint space of the network
    """

    # TODO think about ways we can use the symmetry of the robots to prevent
    # checking redundant edges

    def __init__(self, robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN):
        self._ROADMAP_TYPE = self.__class__.__name__
        self._N_SAMPLE = N_SAMPLE

        super().__init__(robots, env, goalLocs, N_SAMPLE, N_KNN, MAX_EDGE_LEN)
        file_id = (
            f"{self._env.setting}_{self._robots._start_config}_"
            f"{self._N_SAMPLE}samples_{self._N_KNN}nn_{self._MAX_EDGE_LEN}"
            f"len_{self._robots.get_num_robots()}rob_{self._ROADMAP_TYPE}"
        )
        cwd = os.getcwd()
        self._roadmap_filename = f"{cwd}/cached_planning/roadmap_{file_id}.txt"
        self._sample_locs_filename = f"{cwd}/cached_planning/sample_locs_{file_id}.txt"

        # keep track of cost of state-to-state transitions
        self._edge_costs = {}

        # keep track of states which will never be valid due to rigidity
        self._nonrigid_states = set()

        super().init_sample_locs_and_roadmap()

        print(f"Init RandomRoadmap. FileID: {self._roadmap_filename}")

    def generate_sample_locs(self) -> List[List]:
        """Generates a list of predetermined number of deterministic but
        randomly sampled locations without checking if they are in free
        space. Sampling is according to the halton distribution. This also
        adds the start locations and goal locations of the planner to the
        list. Then returns the list

        Returns: List[List]: A list of the sample locations. Each second list is
            n (x,y) pair
        """
        xlb, xub, ylb, yub = self._env._bounds
        sample_locs = []
        distribution = chaospy.J(
            chaospy.Uniform(xlb + 0.1, xub - 0.1), chaospy.Uniform(ylb + 0.1, yub - 0.1)
        )
        sample_locs = distribution.sample(self._N_SAMPLE, rule="halton").transpose()
        assert sample_locs.shape == (
            self._N_SAMPLE,
            2,
        ), f"Sampled locations not right shape - current shape: {sample_locs.shape}, expected shape: ({self._N_SAMPLE}, 2)"

        for loc in self._start_loc_list:
            print(list(loc))
            sample_locs = np.vstack((sample_locs, loc))
        for loc in self._goal_locs:
            sample_locs = np.vstack((sample_locs, loc))

        assert sample_locs.shape == (
            self._N_SAMPLE + 2 * len(self._start_loc_list),
            2,
        ), f"Sampled locations not right shape - current shape: {sample_locs.shape}, expected shape: ({self._N_SAMPLE + 2*len(self._start_loc_list)}, 2)"

        return sample_locs

    def is_valid_path(self, curLoc, connectingLoc):
        """All paths are valid during construction so return True for
        everything. Overwrites parent method which checks for this

        The parent class uses this function to check if edges will run a path
        through an obstacle while building the underlying 2-D roadmap.
        Because right off the bat we don't care about obstacles in our case
        we just ignore this

        Args:
            curLoc (list): location coords of starting node
            connectingLoc (list): location coords of end node

        Returns:
            True
        """
        # TODO test out results when we first check for collisions with obstacles

        dx = curLoc[0] - connectingLoc[0]
        dy = curLoc[1] - connectingLoc[1]
        dist = math.hypot(dx, dy)
        # node too far away
        return dist <= self._MAX_EDGE_LEN

    def set_edge_cost(self, edge: MetaEdge, cost: float):
        """sets a cost to the associated edge

        Args:
            edge (MetaEdge): edge between two MetaStates
            cost (float): cost associated with edge
        """
        self._edge_costs[edge] = cost

    def set_nonrigid_state(self, state: MetaState):
        self._nonrigid_states.add(state)

    def get_state_locations(self, state: MetaState) -> List[List[float]]:
        loc_ids = state.get_loc_ids()
        pos = [self.get_loc(loc_id) for loc_id in loc_ids]
        return pos

    def calc_dist_between_states(self, state1: MetaState, state2: MetaState) -> float:
        """Treats the distance between MetaStates as the square root of the sum
        of the squared differences of each of the corresponding coordinates.
        This is slightly different than the sum of the distance that each robot
        will travel.

        dist = sqrt( sum( (x_i - x_j)**2 ) )

        Args:
            state1 (MetaState): one state to compare with (ordering arbitrary)
            state2 (MetaState): other state to find the distance with

        Returns:
            float: the distance as described above
        """

        # should use (num_robots) x 2 matrices to represent the locations of each
        # network and then take the frobenius norm of the difference of the two

        pos1 = np.array(self.get_state_locations(state1))
        pos2 = np.array(self.get_state_locations(state2))

        diff = pos1 - pos2
        dist = np.linalg.norm(diff, ord="fro")
        return dist

    def get_connecting_metastates(self, state: MetaState) -> List[MetaState]:
        """Returns all MetaStates which are reachable from a given MetaState

        Args:
            state (MetaState): the MetaState to expand from

        Returns:
            List[MetaState]: set of all reachable MetaStates which we haven't
                already found to be nonrigid
        """
        loc_ids = state.get_loc_ids()
        connection_list = []
        for loc_id in loc_ids:
            connections = self.get_connections(loc_id)
            connection_list.append(connections)

        # get all possible connections between the metastates
        connecting_metastate_loc_ids = set(itertools.product(*connection_list))

        meta_states = set()

        for loc_ids in connecting_metastate_loc_ids:
            state = MetaState(self._robots.get_num_robots(), loc_ids)

            # need to make sure we don't return any states already marked not rigid
            if state not in self._nonrigid_states:
                meta_states.add(state)

        return meta_states
