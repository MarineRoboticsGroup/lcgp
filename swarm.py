import time
import numpy as np

import graph
import math_utils


class Swarm:
    def __init__(self, sensingRadius, noise_model, noise_stddev):
        self.sensingRadius = sensingRadius

        assert(noise_model == 'add' or noise_model == 'lognorm')
        self.noise_model = noise_model
        self.noise_stddev = noise_stddev
        self.robot_graph = graph.Graph(self.noise_model, self.noise_stddev)

    """ Swarm Utils """

    def initialize_swarm(self, env, bounds, formation='square', nRobots=None, min_eigval=0.75):
        # initialize formation and edges
        self.startConfig = formation
        self.min_eigval = min_eigval
        self.robot_graph.remove_all_nodes()
        self.robot_graph.remove_all_edges()

        if formation.lower() == 'square':
            self.robot_graph.init_square_formation()
        elif formation.lower() == 'test6':
            self.robot_graph.init_test6_formation()
        elif formation.lower() == 'test8':
            self.robot_graph.init_test8_formation()
        elif formation.lower() == 'test20':
            self.robot_graph.init_test20_formation()
        elif formation.lower() == 'random':
            self.robot_graph.init_random_formation(env, nRobots, bounds)
        else:
            print("The given formation is not valid\n")
            raise NotImplementedError

        # self.reorder_robots()
        self.update_swarm()
        self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    def initialize_swarm_from_loc_list_of_tuples(self, loc_list):
        self.robot_graph.remove_all_nodes()
        self.robot_graph.remove_all_edges()
        for loc in loc_list:
            self.robot_graph.add_node(loc[0], loc[1])
        self.update_swarm()
        if self.robot_graph.get_num_edges() > 0:
            self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    def initialize_swarm_from_loc_list(self, loc_list):
        assert(len(loc_list) % 2 == 0)
        self.robot_graph.remove_all_nodes()
        self.robot_graph.remove_all_edges()

        for i in range(int(len(loc_list)/2)):
            self.robot_graph.add_node(loc_list[2*i], loc_list[2*i+1])

        self.update_swarm()
        self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    def reorder_robots(self):
        raise NotImplementedError

    def update_swarm(self):
        self.robot_graph.update_edges_by_radius(self.sensingRadius)
        if self.robot_graph.get_num_edges() > 0:
            self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    """ Accessors """

    def get_sensing_radius(self):
        return self.sensingRadius

    def get_noise_stddev(self):
        return self.noise_stddev

    def get_noise_model(self):
        return self.noise_model

    def get_robot_graph(self):
        return self.robot_graph

    def get_num_robots(self):
        return self.robot_graph.get_num_nodes()

    def get_position_list_tuples(self):
        return self.robot_graph.get_node_loc_list()

    def get_position_list(self):
        posList = []
        for loc in self.robot_graph.get_node_loc_list():
            posList += list(loc)
        return posList

    def get_nth_eigval(self, n):
        eigvals = math_utils.get_list_all_eigvals(self.fisher_info_matrix)
        eigvals.sort()
        return eigvals[n-1]

    def get_nth_eigpair(self, n):
        eigpair = math_utils.get_nth_eigpair(self.fisher_info_matrix, n)
        return eigpair

    """ Computation """

    def get_gradient_nth_eigval(self, n):
        nthEigenpair = math_utils.get_nth_eigpair(self.fisher_info_matrix, n)
        if not (nthEigenpair[0] > 0):
            return False

        gradient = math_utils.get_gradient_of_eigpair(
            self.fisher_info_matrix, nthEigenpair, self.robot_graph)
        return gradient

    """ Checks """

    def test_rigidity_from_loc_list(self, loc_list):
        if len(loc_list) < 3:
            return False
        test_graph = graph.Graph(self.noise_model, self.noise_stddev)
        test_graph.initialize_from_location_list(loc_list, self.sensingRadius)
        eigval = test_graph.get_nth_eigval(4)
        return (self.min_eigval <= eigval)

    def is_swarm_rigid(self):
        eigval = self.get_nth_eigval(4)
        return (not (eigval == 0))

    """ Control """

    def move_swarm(self, vector, is_relative_move=True):
        self.robot_graph.move_to(vector, is_relative_move=is_relative_move)

    """ Display Utils """

    def print_fisher_matrix(self):
        math_utils.matprint_block(self.fisher_info_matrix)

    def print_all_eigvals(self):
        eigvals = math_utils.get_list_all_eigvals(self.fisher_info_matrix)
        eigvals.sort()
        print(eigvals)

    def print_nth_eigvals(self, n):
        eigvals = math_utils.get_list_all_eigvals(self.fisher_info_matrix)
        eigvals.sort()
        print(eigvals[n-1])

    def show_swarm(self):
        raise NotImplementedError
        # self.robot_graph.displayGraphWithEdges()
