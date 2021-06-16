"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)
modified: alanpapalia

"""

import graph

import math
import collections

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la


class DecoupledRRT:
    """
    Class for RRT planning
    """

    class RRTNode(graph.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.parent = None

        def setParent(parentNode):
            self.parent = parentNode

    def __init__(
        self,
        robot_graph,
        goal_locs,
        obstacle_list,
        bounds,
        max_move_dist=2.0,
        goal_sample_rate=5,
        max_iter=2000,
    ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.nRobots = robot_graph.get_num_nodes()
        start_loc_list = robot_graph.get_node_loc_list()
        self.start = [self.RRTNode(loc[0], loc[1]) for loc in start_loc_list]
        self.end = [self.RRTNode(loc[0], loc[1]) for loc in goal_locs]
        self._bounds = bounds
        self.max_move_dist = max_move_dist
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.found_goal = [False for i in range(self.nRobots)]

    def planning(self):
        print("Beginning Decoupled RRT Planner")
        for i, startPos in enumerate(self.start):
            print(
                "Starting Location:",
                startPos.get_loc_tuple(),
                "End Goal:",
                self.end[i].get_loc_tuple(),
            )

        self.node_list = [[startLoc] for startLoc in self.start]
        for i in range(self.max_iter):
            newNodes = []
            for robotIndex in range(self.nRobots):
                if not self.hasFoundGoal(robotIndex):
                    randomNode = self.generateRandomNode(robotIndex)
                    nearestNode = self.getNearestNode(randomNode, robotIndex)
                    newNode = self.steer(nearestNode, randomNode)

                    if self.isInFreeSpace(newNode):
                        self.node_list[robotIndex].append(newNode)

                        if (
                            self.getDistanceToGoal(newNode, robotIndex)
                            < self.max_move_dist
                        ):
                            self.end[robotIndex].parent = newNode
                            self.found_goal[robotIndex] = True

            if self.allRobotsFoundGoal():
                return self.getListOfFinalTrajectories()

        return None  # cannot find path

    def steer(self, from_node, to_node):
        xFrom, yFrom = from_node.get_loc_tuple()
        xTo, yTo = to_node.get_loc_tuple()

        newNode = self.RRTNode(xFrom, yFrom)

        if self.getDistanceBetweenNodes(newNode, to_node) < self.max_move_dist:
            newNode.move_node_absolute(xTo, yTo)
        else:
            delta = np.array([xTo - xFrom, yTo - yFrom])
            delta = self.max_move_dist * (delta / la.norm(delta, 2))
            newNode.move_node_relative(delta[0], delta[1])

        newNode.parent = from_node
        return newNode

    def getListOfFinalTrajectories(self):
        trajs = []
        for robotIndex in range(self.nRobots):
            trajs.append(self.getFinalTrajectory(robotIndex))
        return trajs

    def getFinalTrajectory(self, robotIndex):
        traj = collections.deque()
        node = self.end[robotIndex]
        while node.parent is not None:
            traj.appendleft(node.get_loc_tuple())
            node = node.parent
        traj.appendleft(node.get_loc_tuple())
        return list(traj)

    def getDistanceToGoal(self, node, index):
        x, y = node.get_loc_tuple()
        xend, yend = self.end[index].get_loc_tuple()
        dx = x - xend
        dy = y - yend
        return math.hypot(dx, dy)

    def generateRandomNode(self, index):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            xlb, xub, ylb, yub = self._bounds
            rnd = self.RRTNode(np.random.uniform(xlb, xub), np.random.uniform(ylb, yub))
        else:  # goal point sampling
            x, y = self.end[index].get_loc_tuple()
            rnd = self.RRTNode(x, y)
        return rnd

    def hasFoundGoal(self, index):
        return self.found_goal[index]

    def allRobotsFoundGoal(self):
        for res in self.found_goal:
            if not res:
                return False
        return True

    def getNearestNode(self, origNode, index):
        candidateNodes = self.node_list[index]
        dlist = [
            self.getDistanceBetweenNodes(origNode, otherNode)
            for otherNode in candidateNodes
        ]
        minind = dlist.index(min(dlist))
        return self.node_list[index][minind]

    def isInFreeSpace(self, node):
        if node is None:
            return False

        coords = node.get_loc_tuple()
        for obs in self.obstacle_list:
            if obs.is_inside(coords):
                return False

        return True  # safe

    def getDistanceBetweenNodes(self, from_node, to_node):
        xFrom, yFrom = from_node.get_loc_tuple()
        xTo, yTo = to_node.get_loc_tuple()

        dx = xTo - xFrom
        dy = yTo - yFrom
        d = math.hypot(dx, dy)
        return d
