import sys
sys.path.insert(1, './snl')

import numpy as np
from numpy import linalg as la
from scipy.linalg import null_space, toeplitz

import math_utils
import environment
import kdtree
from snl import SolveSNLWithSDP

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

    def PerformSNL(self, init_guess, solver:str=None):
        num_anchors = 3
        num_nodes = self.getNumNodes() - num_anchors
        anchor_ids = [v+num_nodes for v in range(num_anchors)]
        anchor_locs = {}
        for id in anchor_ids:
            anchor_locs[id] = self.getNodePositionTuple(id)
        node_node_dists = {}
        node_anchor_dists = {}
        for edge in self.getGraphEdgeList():
            i, j = edge
            dist = self.getEdgeDistScal(edge)
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

        loc_est = SolveSNLWithSDP(num_nodes, node_node_dists, node_anchor_dists, anchor_locs, anchor_ids, init_guess=init_guess, use_spring_solver=use_spring_solver)
        return loc_est

    ###### Initialize and Format Graph ########
    def initializeFromLocationList(self, locationList, radius):
        self.removeAllNodes()
        for loc in locationList:
            self.addNode(loc[0], loc[1])
        self.updateEdgesByRadius(radius)

    def addNode(self, xLoc, yLoc):
        self.nodes.append(Node(xLoc, yLoc))
        self.nNodes += 1

    def addGraphEdge(self, node1, node2):
        assert(self.nodeExists(node1))
        assert(self.nodeExists(node2))

        edge = (node1, node2)
        if not (self.edgeExists(edge)):
            self.edges.append(edge)
            self.edgeDistances.append(self.getEdgeDistScal(edge))
            self.nEdges += 1
            self.nodes[node1].addNodeEdge(node2)
            self.nodes[node2].addNodeEdge(node1)

    def removeGraphNode(self, nodeNum):
        assert(self.nodeExists(nodeNum))
        self.removeConnectingNodeEdges(nodeNum)
        self.nodes.remove(nodeNum)
        self.nNodes -= 1

    def removeGraphEdge(self, edge):
        assert(self.edgeExists(edge))
        self.nEdges -= 1
        n1, n2 = edge
        if (n1,n2) in self.edges:
            self.edges.remove((n1, n2))
        else:
            self.edges.remove((n2, n1))
        self.nodes[n1].removeNodeEdge(n2)
        self.nodes[n2].removeNodeEdge(n1)

    def removeAllNodes(self, ):
        self.removeAllEdges()
        self.nodes.clear()
        self.nNodes = 0

    def removeAllEdges(self, ):
        self.nEdges = 0
        self.edges.clear()
        self.edgeDistances.clear()

    def updateEdgesByRadius(self, radius):
        self.removeAllEdges()
        if self.nNodes <= 1:
            return
        nodeKDTree = kdtree.KDTree(self.getNodeLocationList())
        self.nEdges = 0
        for r1 in range(self.nNodes):
            curLoc = self.nodes[r1].getXYLocation()
            index, dists = nodeKDTree.search(np.array(curLoc).reshape(2, 1), k=self.nNodes)
            inds = index[0][1:]
            ds = dists[0][1:]
            for r2, d2 in zip(inds, ds):
                if d2 < radius:
                    self.addGraphEdge(r1, r2)

    def removeConnectingNodeEdges(self, nodeNum):
        assert(self.nodeExists(nodeNum))
        edgeList = self.getListOfNodeEdgePairs(nodeNum).copy()
        for connection in edgeList:
            self.removeGraphEdge(connection)
            self.nEdges -= 1

    ###### Graph Accessors ########
    def getGraphEdgeList(self, ):
        return self.edges

    def getGraphNodeList(self, ):
        return self.nodes

    def getNumNodes(self, ):
        return self.nNodes

    def getNumEdges(self, ):
        return self.nEdges

    def getNthEigval(self, n):
        eigvals = math_utils.getListOfAllEigvals(self.getStiffnessMatrix())
        eigvals.sort()
        return eigvals[n-1]

    def getStiffnessMatrix(self, ):
        n = self.nNodes
        K = np.zeros((2*n, 2*n))
        return math_utils.buildStiffnessMatrix(self.edges, self.nodes, self.noise_model, self.noise_stddev)

    def getNodeLocationList(self, ):
        locs = []
        for node in self.nodes:
            locs.append(node.getXYLocation())
        return locs

    ###### Node Accessors ########
    def getNodePositionTuple(self, nodeNum):
        assert (self.nodeExists(nodeNum))
        node = self.nodes[nodeNum]
        return node.getXYLocation()

    def getNodeDegree(self, nodeNum):
        assert (nodeExists(n))
        node = self.nodes[nodeNum]
        return node.getNodeDegree()

    def getNodeConnectionList(self, nodeNum):
        node = self.nodes[nodeNum]
        return node.getNodeConnections()

    def getListOfNodeEdgePairs(self, nodeNum):
        assert(self.nodeExists(nodeNum))
        node = self.nodes[nodeNum]
        edges = []
        for node2 in node.getNodeConnections():
            edge = (nodeNum, node2)
            edges.append(edge)
        return edges

    def getEdgeDistVec(self, edge):
        assert(self.edgeExists(edge))
        id1, id2 = edge
        node1 = self.nodes[id1]
        node2 = self.nodes[id2]
        x1, y1 = node1.getXYLocation()
        x2, y2 = node2.getXYLocation()
        v = np.array([x1-x2, y1-y2])
        return v

    def getEdgeDistScal(self, edge):
        assert(self.edgeExists(edge))
        v = self.getEdgeDistVec(edge)
        return la.norm(v, 2)

    def getDistVecBetweenNodes(self, n1 , n2):
        assert(self.nodeExists(n1))
        assert(self.nodeExists(n2))
        node1 = self.nodes[n1]
        node2 = self.nodes[n2]
        x1, y1 = node1.getXYLocation()
        x2, y2 = node2.getXYLocation()
        v = np.array([x1-x2, y1-y2])
        return v

    def getDistBetweenNodes(self, n1 , n2):
        assert(self.nodeExists(n1))
        assert(self.nodeExists(n2))
        v = self.getDistVecBetweenNodes(n1, n2)
        dist = la.norm(v, 2)
        return dist

    ####### Construct Graph Formations #######
    def initializeTest6(self):
        self.addNode(0, 0)
        self.addNode(0, 2)
        self.addNode(1, 3)
        self.addNode(2, 2)
        self.addNode(2, 0)
        self.addNode(1, 1)

    def initializeTest8(self):
        self.addNode(2, 2)
        self.addNode(2, 4)
        self.addNode(4, 2)
        self.addNode(4, 4)
        self.addNode(6, 4)
        self.addNode(6, 6)
        self.addNode(8, 6)
        self.addNode(8, 8)

    def initializeSquare(self):
        # self.addNode(1, 1)
        # self.addNode(1, 2)
        # self.addNode(2, 2)
        # self.addNode(2, 1)
        self.addNode(2, 2)
        self.addNode(2, 6)
        self.addNode(6, 6)
        self.addNode(6, 2)

    def initializeRandomConfig(self, numRobots, bounds):
        xbound, ybound = bounds
        xVal = np.random.uniform(low=1, high=10, size=numRobots)
        yVal = np.random.uniform(low=1, high=10, size=numRobots)
        for i in range(numRobots):

            self.addNode(xVal[i], yVal[i])

    ####### Controls #######
    def moveTowardsVec(self, vec, relativeMovement=True):
        assert(len(vec) == 2*len(self.nodes))
        for i, node in enumerate(self.nodes):
            newX = vec[2*i]
            newY = vec[2*i+1]

            if relativeMovement:
                node.moveNodeRelative(newX, newY)
            else:
                node.moveNodeAbsolute(newX, newY)

    ####### Testing #######
    def edgeExists(self, edge):
        n1, n2 = edge[0], edge[1]
        assert(self.nodeExists(n1))
        assert(self.nodeExists(n2))
        return (((n1, n2) in self.edges ) or ((n2, n1) in self.edges ))

    def nodeExists(self, node):
        return (node < self.nNodes)

class Node:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._degree = 0
        self._connections = []

    def getNodeConnections(self, ):
        return self._connections

    def getNodeDegree(self, ):
        return self._degree

    def getXYLocation(self, ):
        return (self._x, self._y)

    def addNodeEdge(self, edgeNodeNum):
        self._degree += 1
        self._connections.append(edgeNodeNum)

    def removeNodeEdge(self, edgeNodeNum):
        self._degree -= 1
        self._connections.remove(edgeNodeNum)

    def moveNodeAbsolute(self, newX, newY):
        self._x = newX
        self._y = newY

    def moveNodeRelative(self, deltaX, deltaY):
        self._x += deltaX
        self._y += deltaY
