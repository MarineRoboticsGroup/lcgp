import sys
sys.path.insert(1, './snl')

import numpy as np
from numpy import linalg as la
from scipy.linalg import null_space, toeplitz

import math_utils
import environment
import kdtree
from snl import solve_snl_with_sdp

class Graph:
    def __init__(self, noise_model:str, noise_stddev:float):
        self.edges = []
        self.edgeDistances = []
        self.nodes = []
        self.nNodes = 0
        self.nEdges = 0

        assert(noise_model == 'add' or noise_model == 'lognorm')
        self.noise_model = noise_model
        self.noise_stddev = noise_stddev
        self.fisher_info_matrix = None

    def perform_snl(self, init_guess=None, solver:str=None):
        num_anchors = 3
        num_nodes = self.get_num_nodes() - num_anchors
        anchor_ids = [v+num_nodes for v in range(num_anchors)]
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
                noisy_dist = dist+noise
            elif self.noise_model == "lognorm":
                noise = np.random.normal(1, self.noise_stddev)
                noisy_dist = dist*noise
            else:
                raise NotImplementedError

            if i in anchor_ids and j in anchor_ids:
                continue
            elif i in anchor_ids or j in anchor_ids:
                node_anchor_dists[edge] = noisy_dist
            else:
                node_node_dists[edge] = noisy_dist

        loc_est = solve_snl_with_sdp(num_nodes, node_node_dists, node_anchor_dists, anchor_locs, anchor_ids, init_guess=init_guess, solver=solver)
        return loc_est

    ###### Initialize and Format Graph ########
    def initialize_from_location_list(self, locationList, radius):
        self.remove_all_nodes()
        for loc in locationList:
            self.add_node(loc[0], loc[1])
        self.update_edges_by_radius(radius)

    def add_node(self, xLoc, yLoc):
        self.nodes.append(Node(xLoc, yLoc))
        self.nNodes += 1

    def add_graph_edge(self, node1, node2):
        assert(self.node_exists(node1))
        assert(self.node_exists(node2))

        edge = (node1, node2)
        if not (self.edge_exists(edge)):
            self.edges.append(edge)
            self.edgeDistances.append(self.get_edge_dist_scal(edge))
            self.nEdges += 1
            self.nodes[node1].add_node_edge(node2)
            self.nodes[node2].add_node_edge(node1)

    def remove_graph_node(self, nodeNum):
        assert(self.node_exists(nodeNum))
        self.remove_connecting_node_edges(nodeNum)
        self.nodes.remove(nodeNum)
        self.nNodes -= 1

    def remove_graph_edge(self, edge):
        assert(self.edge_exists(edge))
        self.nEdges -= 1
        n1, n2 = edge
        if (n1,n2) in self.edges:
            self.edges.remove((n1, n2))
        else:
            self.edges.remove((n2, n1))
        self.nodes[n1].remove_node_edge(n2)
        self.nodes[n2].remove_node_edge(n1)

    def remove_all_nodes(self, ):
        self.remove_all_edges()
        self.nodes.clear()
        self.nNodes = 0

    def remove_all_edges(self, ):
        self.nEdges = 0
        self.edges.clear()
        self.edgeDistances.clear()

    def update_edges_by_radius(self, radius):
        self.remove_all_edges()
        if self.nNodes <= 1:
            return
        self.nEdges = 0
        for id1 in range(self.nNodes):
            for id2 in range(id1+1, self.nNodes):
                dist = self.get_dist_scal_between_nodes(id1, id2)
                if dist < radius:
                    self.add_graph_edge(id1, id2)

    def remove_connecting_node_edges(self, nodeNum):
        assert(self.node_exists(nodeNum))
        edgeList = self.get_list_of_node_edge_pairs(nodeNum).copy()
        for connection in edgeList:
            self.remove_graph_edge(connection)
            self.nEdges -= 1

    ###### Graph Accessors ########
    def get_graph_edge_list(self, ):
        return self.edges

    def get_graph_node_list(self, ):
        return self.nodes

    def get_num_nodes(self, ):
        return self.nNodes

    def get_num_edges(self, ):
        return self.nEdges

    def get_nth_eigval(self, n):
        eigvals = math_utils.get_list_all_eigvals(self.get_fisher_matrix())
        eigvals.sort()
        return eigvals[n-1]

    def get_fisher_matrix(self, ):
        return math_utils.build_fisher_matrix(self.edges, self.nodes, self.noise_model, self.noise_stddev)

    def get_node_loc_list(self, ):
        locs = []
        for node in self.nodes:
            locs.append(node.get_loc_tuple())
        return locs

    ###### Node Accessors ########
    def get_node_loc_tuple(self, nodeNum):
        assert (self.node_exists(nodeNum))
        node = self.nodes[nodeNum]
        return node.get_loc_tuple()

    def get_node_degree(self, nodeNum):
        assert (self.node_exists(nodeNum))
        node = self.nodes[nodeNum]
        return node.get_node_degree()

    def get_node_connection_list(self, nodeNum):
        node = self.nodes[nodeNum]
        return node.get_node_connections()

    def get_list_of_node_edge_pairs(self, nodeNum):
        assert(self.node_exists(nodeNum))
        node = self.nodes[nodeNum]
        edges = []
        for node2 in node.get_node_connections():
            edge = (nodeNum, node2)
            edges.append(edge)
        return edges

    def get_edge_dist_vec(self, edge):
        assert(self.edge_exists(edge))
        id1, id2 = edge
        node1 = self.nodes[id1]
        node2 = self.nodes[id2]
        x1, y1 = node1.get_loc_tuple()
        x2, y2 = node2.get_loc_tuple()
        v = np.array([x1-x2, y1-y2])
        return v

    def get_edge_dist_scal(self, edge):
        assert(self.edge_exists(edge))
        v = self.get_edge_dist_vec(edge)
        return la.norm(v, 2)

    def get_dist_vec_between_nodes(self, n1 , n2):
        assert(self.node_exists(n1))
        assert(self.node_exists(n2))
        node1 = self.nodes[n1]
        node2 = self.nodes[n2]
        x1, y1 = node1.get_loc_tuple()
        x2, y2 = node2.get_loc_tuple()
        v = np.array([x1-x2, y1-y2])
        return v

    def get_dist_scal_between_nodes(self, n1 , n2):
        assert(self.node_exists(n1))
        assert(self.node_exists(n2))
        v = self.get_dist_vec_between_nodes(n1, n2)
        dist = la.norm(v, 2)
        return dist

    ####### Construct Graph Formations #######
    def init_test6_formation(self):
        self.add_node(0, 0)
        self.add_node(0, 2)
        self.add_node(1, 3)
        self.add_node(2, 2)
        self.add_node(2, 0)
        self.add_node(1, 1)

    def init_test8_formation(self):
        self.add_node(2, 2)
        self.add_node(2, 4)
        self.add_node(4, 2)
        self.add_node(4, 4)
        self.add_node(6, 4)
        self.add_node(6, 6)
        self.add_node(8, 6)
        self.add_node(8, 8)

    def init_square_formation(self):
        # self.add_node(1, 1)
        # self.add_node(1, 2)
        # self.add_node(2, 2)
        # self.add_node(2, 1)
        self.add_node(2, 2)
        self.add_node(2, 6)
        self.add_node(6, 6)
        self.add_node(6, 2)

    def init_random_formation(self, num_robots, bounds):
        xVal = np.random.uniform(low=1, high=10, size=num_robots)
        yVal = np.random.uniform(low=1, high=10, size=num_robots)
        for i in range(num_robots):

            self.add_node(xVal[i], yVal[i])

    ####### Controls #######
    def move_to(self, vec, is_relative_move=True):
        assert(len(vec) == 2*len(self.nodes))
        for i, node in enumerate(self.nodes):
            newX = vec[2*i]
            newY = vec[2*i+1]

            if is_relative_move:
                node.move_node_relative(newX, newY)
            else:
                node.move_node_absolute(newX, newY)

    ####### Testing #######
    def edge_exists(self, edge):
        n1, n2 = edge[0], edge[1]
        assert(self.node_exists(n1))
        assert(self.node_exists(n2))
        return (((n1, n2) in self.edges ) or ((n2, n1) in self.edges ))

    def node_exists(self, node):
        return (node < self.nNodes)

class Node:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._degree = 0
        self._connections = []

    def get_node_connections(self, ):
        return self._connections

    def get_node_degree(self, ):
        return self._degree

    def get_loc_tuple(self, ):
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
