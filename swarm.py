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

    ####### Swarm Utils #######
    def initializeSwarm(self, bounds, formation='square', nRobots=None, minEigval=0.75):
        # intialize formation and edges
        self.startConfig = formation
        self.minEigval = minEigval
        self.robot_graph.remove_all_nodes()
        self.robot_graph.remove_all_edges()

        if formation.lower() == 'square':
            self.robot_graph.init_square_formation()
        elif formation.lower() == 'test6':
            self.robot_graph.init_test6_formation()
        elif formation.lower() == 'test8':
            self.robot_graph.init_test8_formation()
        elif formation.lower() == 'random':
            self.robot_graph.init_random_formation(nRobots, bounds)
        else:
            print("The given formation is not valid\n")
            raise NotImplementedError

        # self.reorderRobotsBasedOnConnectivity()
        self.updateSwarm()
        self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    def initializeSwarmFromLocationListTuples(self, locList):
        self.robot_graph.remove_all_nodes()
        self.robot_graph.remove_all_edges()
        for loc in locList:
            self.robot_graph.add_node(loc[0], loc[1])
        self.updateSwarm()
        if self.robot_graph.get_num_edges() > 0:
            self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    def initializeSwarmFromLocationList(self, locList):
        assert(len(locList)%2 == 0)
        self.robot_graph.remove_all_nodes()
        self.robot_graph.remove_all_edges()

        for i in range(int(len(locList)/2)):
            self.robot_graph.add_node(locList[2*i], locList[2*i+1])

        self.updateSwarm()
        self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    def reorderRobotsBasedOnConnectivity(self):
        raise NotImplementedError

    def updateSwarm(self):
        self.robot_graph.update_edges_by_radius(self.sensingRadius)
        if self.robot_graph.get_num_edges() > 0:
            self.fisher_info_matrix = self.robot_graph.get_fisher_matrix()

    ####### Accessors #######
    def getSensingRadius(self):
        return self.sensingRadius

    def getRobotGraph(self):
        return self.robot_graph

    def getNumRobots(self):
        return self.robot_graph.get_num_nodes()

    def get_position_list_tuples(self):
        return self.robot_graph.get_node_loc_list()

    def getPositionList(self):
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

    ####### Computation #######
    def getGradientOfNthEigenval(self, n):
        nthEigenpair = math_utils.getEigpairLeastFirst(self.fisher_info_matrix, n-1)
        if not (nthEigenpair[0] > 0):
            return False

        gradient = math_utils.get_gradient_of_eigpair(self.fisher_info_matrix, nthEigenpair, self.robot_graph)
        return gradient

    ####### Checks #######
    def testRigidityFromLocList(self, locList):
        testGraph = graph.Graph(self.noise_model, self.std_dev)
        testGraph.initialize_from_location_list(locList, self.sensingRadius)
        eigval = testGraph.get_nth_eigval(4)
        return (self.minEigval <= eigval)

    def isRigidFormation(self):
        eigval = self.get_nth_eigval(4)
        return (not (eigval == 0))

    ####### Control #######
    def moveSwarm(self, vector, moveRelative=True):
        self.robot_graph.move_to(vector, relativeMovement=moveRelative)

    ####### Display Utils #######
    def printStiffnessMatrix(self):
        math_utils.matprint_block(self.fisher_info_matrix)
    def printAllEigvals(self):
        eigvals = math_utils.get_list_all_eigvals(self.fisher_info_matrix)
        eigvals.sort()
        print(eigvals)

    def printNthEigval(self, n):
        eigvals = math_utils.get_list_all_eigvals(self.fisher_info_matrix)
        eigvals.sort()
        print(eigvals[n-1])
    def showSwarm(self):
        self.robot_graph.displayGraphWithEdges()
