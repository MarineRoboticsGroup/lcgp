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
        self.robot_graph.removeAllNodes()
        self.robot_graph.removeAllEdges()

        if formation.lower() == 'square':
            self.robot_graph.initializeSquare()
        elif formation.lower() == 'test6':
            self.robot_graph.initializeTest6()
        elif formation.lower() == 'test8':
            self.robot_graph.initializeTest8()
        elif formation.lower() == 'random':
            self.robot_graph.initializeRandomConfig(nRobots, bounds)
        else:
            print("The given formation is not valid\n")
            raise NotImplementedError

        # self.reorderRobotsBasedOnConnectivity()
        self.updateSwarm()
        self.fisher_info_matrix = self.robot_graph.getStiffnessMatrix()

    def initializeSwarmFromLocationListTuples(self, locList):
        self.robot_graph.removeAllNodes()
        self.robot_graph.removeAllEdges()
        for loc in locList:
            self.robot_graph.addNode(loc[0], loc[1])
        self.updateSwarm()
        if self.robot_graph.getNumEdges() > 0:
            self.fisher_info_matrix = self.robot_graph.getStiffnessMatrix()

    def initializeSwarmFromLocationList(self, locList):
        assert(len(locList)%2 == 0)
        self.robot_graph.removeAllNodes()
        self.robot_graph.removeAllEdges()

        for i in range(int(len(locList)/2)):
            self.robot_graph.addNode(locList[2*i], locList[2*i+1])

        self.updateSwarm()
        self.fisher_info_matrix = self.robot_graph.getStiffnessMatrix()

    def reorderRobotsBasedOnConnectivity(self):
        raise NotImplementedError

    def updateSwarm(self):
        self.robot_graph.updateEdgesByRadius(self.sensingRadius)
        if self.robot_graph.getNumEdges() > 0:
            self.fisher_info_matrix = self.robot_graph.getStiffnessMatrix()

    ####### Accessors #######
    def getSensingRadius(self):
        return self.sensingRadius

    def getRobotGraph(self):
        return self.robot_graph

    def getNumRobots(self):
        return self.robot_graph.getNumNodes()

    def getPositionListTuples(self):
        return self.robot_graph.getNodeLocationList()

    def getPositionList(self):
        posList = []
        for loc in self.robot_graph.getNodeLocationList():
            posList += list(loc)
        return posList

    def getNthEigval(self, n):
        eigvals = math_utils.getListOfAllEigvals(self.fisher_info_matrix)
        eigvals.sort()
        return eigvals[n-1]
    def getNthEigpair(self, n):
        eigpair = math_utils.getNthEigpair(self.fisher_info_matrix, n)
        return eigpair

    ####### Computation #######
    def getGradientOfNthEigenval(self, n):
        nthEigenpair = math_utils.getEigpairLeastFirst(self.fisher_info_matrix, n-1)
        if not (nthEigenpair[0] > 0):
            return False

        gradient = math_utils.getGradientOfMatrixForEigenpair(self.fisher_info_matrix, nthEigenpair, self.robot_graph)
        return gradient

    ####### Checks #######
    def testRigidityFromLocList(self, locList):
        testGraph = graph.Graph(self.noise_model, self.std_dev)
        testGraph.initializeFromLocationList(locList, self.sensingRadius)
        eigval = testGraph.getNthEigval(4)
        return (self.minEigval <= eigval)

    def isRigidFormation(self):
        eigval = self.getNthEigval(4)
        return (not (eigval == 0))

    ####### Control #######
    def moveSwarm(self, vector, moveRelative=True):
        self.robot_graph.moveTowardsVec(vector, relativeMovement=moveRelative)

    ####### Display Utils #######
    def printStiffnessMatrix(self):
        math_utils.matprintBlock(self.fisher_info_matrix)
    def printAllEigvals(self):
        eigvals = math_utils.getListOfAllEigvals(self.fisher_info_matrix)
        eigvals.sort()
        print(eigvals)

    def printNthEigval(self, n):
        eigvals = math_utils.getListOfAllEigvals(self.fisher_info_matrix)
        eigvals.sort()
        print(eigvals[n-1])
    def showSwarm(self):
        self.robot_graph.displayGraphWithEdges()
