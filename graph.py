from snl import solve_snl_with_sdp
import kdtree
import environment
import math_utils
from scipy.linalg import null_space, toeplitz
from numpy import linalg as la
import numpy as np
import sys
import itertools

sys.path.insert(1, "./snl")


class Graph:
    def __init__(self, noise_model: str, noise_stddev: float):
        self.edges = []
        self.edgeDistances = []
        self.nodes = []
        self.nNodes = 0
        self.nEdges = 0

        assert noise_model == "add" or noise_model == "lognorm"
        self.noise_model = noise_model
        self.noise_stddev = noise_stddev
        self.fisher_info_matrix = None

    def perform_snl(self, init_guess=None, solver: str = None):
        """Uses different sensor network localization techniques

        Args:
            init_guess ([type], optional): [description]. Defaults to None.
            solver (str, optional): The solver type for the SNL. Defaults to None.

        Raises:
            NotImplementedError: Not a valid noise model

        Returns:
            :returns:   Ordered array of estimated locations
            :rtype:     numpy.ndarray, shape = ((num_nodes+num_anchors), 2)
        """
        num_anchors = 3
        num_nodes = self.get_num_nodes() - num_anchors
        if num_nodes > 0:
            anchor_ids = [v + num_nodes for v in range(num_anchors)]
            anchor_locs = {}
            for id in anchor_ids:
                anchor_locs[id] = self.get_node_loc_tuple(id)
            node_node_dists = {}
            node_anchor_dists = {}
            for edge in self.get_graph_edge_list():
                i, j = edge
                dist = self.get_edge_dist_scal(edge)
                if self.noise_model == "add":
                    noise = np.random.normal(0, self.noise_stddev)
                    noisy_dist = dist + noise
                elif self.noise_model == "lognorm":
                    noise = np.random.normal(1, self.noise_stddev)
                    noisy_dist = dist * noise
                else:
                    raise NotImplementedError("Not a valid noise model")

                if i in anchor_ids and j in anchor_ids:
                    continue
                elif i in anchor_ids or j in anchor_ids:
                    node_anchor_dists[edge] = noisy_dist
                else:
                    node_node_dists[edge] = noisy_dist

            loc_est = solve_snl_with_sdp(
                num_nodes,
                node_node_dists,
                node_anchor_dists,
                anchor_locs,
                anchor_ids,
                init_guess=init_guess,
                solver=solver,
            )
            return loc_est
        else:
            anchor_locs = np.array(
                [self.get_node_loc_tuple(i) for i in range(num_anchors)]
            )
            return anchor_locs

    """ Initialize and Format Graph """

    def initialize_from_location_list(self, locationList, radius):
        self.remove_all_nodes()
        for loc in locationList:
            self.add_node(loc[0], loc[1])
        self.update_edges_by_radius(radius)

    def add_node(self, xLoc, yLoc):
        self.nodes.append(Node(xLoc, yLoc))
        self.nNodes += 1

    def add_graph_edge(self, node1, node2):
        assert self.node_exists(node1)
        assert self.node_exists(node2)

        edge = (node1, node2)
        if not (self.edge_exists(edge)):
            self.edges.append(edge)
            self.edgeDistances.append(self.get_edge_dist_scal(edge))
            self.nEdges += 1
            self.nodes[node1].add_node_edge(node2)
            self.nodes[node2].add_node_edge(node1)

    def remove_graph_node(self, nodeNum):
        assert self.node_exists(nodeNum)
        self.remove_connecting_node_edges(nodeNum)
        self.nodes.remove(nodeNum)
        self.nNodes -= 1

    def remove_graph_edge(self, edge):
        assert self.edge_exists(edge)
        self.nEdges -= 1
        n1, n2 = edge
        if (n1, n2) in self.edges:
            self.edges.remove((n1, n2))
        else:
            self.edges.remove((n2, n1))
        self.nodes[n1].remove_node_edge(n2)
        self.nodes[n2].remove_node_edge(n1)

    def remove_all_nodes(
        self,
    ):
        self.remove_all_edges()
        self.nodes.clear()
        self.nNodes = 0

    def remove_all_edges(
        self,
    ):
        self.nEdges = 0
        self.edges.clear()
        self.edgeDistances.clear()

    def update_edges_by_radius(self, radius):
        self.remove_all_edges()
        if self.nNodes <= 1:
            return
        self.nEdges = 0
        for id1 in range(self.nNodes):
            for id2 in range(id1 + 1, self.nNodes):
                dist = self.get_dist_scal_between_nodes(id1, id2)
                if dist < radius:
                    self.add_graph_edge(id1, id2)

    def remove_connecting_node_edges(self, nodeNum):
        assert self.node_exists(nodeNum)
        edgeList = self.get_list_of_node_edge_pairs(nodeNum).copy()
        for connection in edgeList:
            self.remove_graph_edge(connection)
            self.nEdges -= 1

    """ Graph Accessors """

    def get_graph_edge_list(
        self,
    ):
        return self.edges

    def get_graph_node_list(
        self,
    ):
        return self.nodes

    def get_num_nodes(
        self,
    ):
        return self.nNodes

    def get_num_edges(
        self,
    ):
        return self.nEdges

    def get_nth_eigval(self, n, num_anchors):
        eigvals = math_utils.get_list_all_eigvals(self.get_fisher_matrix(num_anchors))
        eigvals.sort()
        return eigvals[n - 1]

    def get_fisher_matrix(self, num_anchors):
        return math_utils.build_fisher_matrix_ungrounded(
            self.edges, self.nodes, self.noise_model, self.noise_stddev, num_anchors
        )

    def get_fisher_matrix_ungrounded(self, num_anchors: int):
        return math_utils.build_fisher_matrix_ungrounded(
            self.edges, self.nodes, self.noise_model, self.noise_stddev, num_anchors
        )

    def get_node_loc_list(
        self,
    ):
        locs = []
        for node in self.nodes:
            locs.append(node.get_loc_tuple())
        return locs

    """ Node Accessors """

    def get_node_loc_tuple(self, nodeNum):
        assert self.node_exists(nodeNum)
        node = self.nodes[nodeNum]
        return node.get_loc_tuple()

    def get_node_degree(self, nodeNum):
        assert self.node_exists(nodeNum)
        node = self.nodes[nodeNum]
        return node.get_node_degree()

    def get_node_connection_list(self, nodeNum):
        node = self.nodes[nodeNum]
        return node.get_node_connections()

    def get_list_of_node_edge_pairs(self, nodeNum):
        assert self.node_exists(nodeNum)
        node = self.nodes[nodeNum]
        edges = []
        for node2 in node.get_node_connections():
            edge = (nodeNum, node2)
            edges.append(edge)
        return edges

    def get_edge_dist_scal(self, edge):
        assert self.edge_exists(edge)
        id1, id2 = edge
        loc1 = self.nodes[id1].get_loc_tuple()
        loc2 = self.nodes[id2].get_loc_tuple()
        return math_utils.calc_dist_between_locations(loc1, loc2)

    def get_dist_scal_between_nodes(self, n1, n2):
        assert self.node_exists(n1)
        assert self.node_exists(n2)
        loc1 = self.nodes[n1].get_loc_tuple()
        loc2 = self.nodes[n2].get_loc_tuple()
        dist = math_utils.calc_dist_between_locations(loc1, loc2)
        return dist

    """ Construct Graph Formations """

    def init_test_simple_vicon_formation(self):
        self.add_node(0.5, 0.9)
        self.add_node(0.5, 1.5)
        self.add_node(1.0, 0.3)
        self.add_node(1.5, 0.9)
        self.add_node(1.5, 1.5)

    def init_anchor_only_test(self):
        self.add_node(3, 3)
        self.add_node(4, 2)
        self.add_node(2, 3)

    def init_test6_formation(self):
        self.add_node(3, 3)
        self.add_node(4, 2)
        self.add_node(2, 3)
        self.add_node(6, 6)
        self.add_node(5, 3)
        self.add_node(2, 6)

    def init_test8_formation(self):
        self.add_node(2, 2)
        self.add_node(2, 4)
        self.add_node(4, 2)
        self.add_node(4, 4)
        self.add_node(6, 4)
        self.add_node(6, 6)
        self.add_node(8, 6)
        self.add_node(8, 8)

    def init_test20_formation(self):
        """Randomly chooses the ordering from a gridded up set of locations"""
        x_range = np.linspace(2, 8, num=4)
        y_range = np.linspace(2, 10, num=5)
        locs = list(itertools.product(x_range, y_range))
        inds = [i for i in range(len(locs))]
        while len(inds) > 0:
            ind = np.random.choice(inds, replace=False)
            inds.remove(ind)
            loc = locs[ind]
            x = loc[0]
            y = loc[1]
            self.add_node(x, y)

    def init_square_formation(self):
        # self.add_node(1, 1)
        # self.add_node(1, 2)
        # self.add_node(2, 2)
        # self.add_node(2, 1)
        self.add_node(2, 2)
        self.add_node(2, 6)
        self.add_node(6, 6)
        self.add_node(6, 2)

    def init_random_formation(self, env, num_robots, bounds):
        loc = math_utils.generate_random_loc(2, 10, 2, 10)
        locs = [loc]

        def distance(loc, ref_loc):
            dist = np.sqrt((loc[0] - ref_loc[0]) ** 2 + (loc[1] - ref_loc[1]) ** 2)
            return dist

        while len(locs) < num_robots:
            loc = math_utils.generate_random_loc(2, bounds[0] / 2, 2, bounds[1] / 2)

            if not env.is_free_space(loc):
                continue

            satisfies_conditions = True
            dists = np.array([distance(loc, existing_loc) for existing_loc in locs])
            if (dists < 1).any():
                satisfies_conditions = False
            elif len(locs) == 1:
                satisfies_conditions = np.count_nonzero(dists < 5) == 1
            elif len(locs) >= 2:
                satisfies_conditions = np.count_nonzero(dists < 5) >= 2

            if satisfies_conditions:
                locs.append(loc)

        for i in range(num_robots):
            self.add_node(locs[i][0], locs[i][1])
        print(f"Randomly Generated Robot Formation")

    """ Controls """

    def move_to(self, vec, is_relative_move=True):
        assert len(vec) == 2 * len(self.nodes)
        for i, node in enumerate(self.nodes):
            newX = vec[2 * i]
            newY = vec[2 * i + 1]

            if is_relative_move:
                node.move_node_relative(newX, newY)
            else:
                node.move_node_absolute(newX, newY)

    """ Testing """

    def edge_exists(self, edge):
        n1, n2 = edge[0], edge[1]
        assert self.node_exists(n1)
        assert self.node_exists(n2)
        return ((n1, n2) in self.edges) or ((n2, n1) in self.edges)

    def node_exists(self, node):
        return node < self.nNodes


class Node:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._degree = 0
        self._connections = []

    def get_node_connections(
        self,
    ):
        return self._connections

    def get_node_degree(
        self,
    ):
        return self._degree

    def get_loc_tuple(
        self,
    ):
        return (self._x, self._y)

    def add_node_edge(self, edgeNodeNum):
        self._degree += 1
        self._connections.append(edgeNodeNum)

    def remove_node_edge(self, edgeNodeNum):
        self._degree -= 1
        self._connections.remove(edgeNodeNum)

    def move_node_absolute(self, newX, newY):
        self._x = newX
        self._y = newY

    def move_node_relative(self, deltaX, deltaY):
        self._x += deltaX
        self._y += deltaY
