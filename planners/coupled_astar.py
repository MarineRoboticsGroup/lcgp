import copy
import string
import math
from itertools import permutations
from os import path
import itertools
import chaospy
import numpy as np
import matplotlib.pyplot as plt

import graph
import kdtree

class CoupledAstar():
    def __init__(self, robots, env, goals, min_eigval=0.75):
        self.robots = robots
        self.env = env
        self.goalLocs = goals
        self.start_loc_list = self.robots.get_position_list_tuples()
        self.min_eigval = min_eigval

        self.num_robots = robots.get_num_robots()

        roadmap_sampling = "uniform"
        self.N_SAMPLE = 200
        self.N_KNN = 4
        self.MAX_EDGE_LEN = 8
        self.roadmap = self.Roadmap(self.robots, self.env, self.goalLocs, roadmap_sampling, self.N_SAMPLE, self.N_KNN, self.MAX_EDGE_LEN)
        self.plot_roadmap()
        self.startIndices = self.roadmap.get_start_indexList()
        self.goalIndices = self.roadmap.get_goal_indexList()
        print("Start Index:", self.startIndices)
        print("Goal Index:", self.goalIndices)

    class CoupledAstarNode:
        def __init__(self, indexList, cost, pind):
            self.indexList = indexList
            self.cost = cost
            self.pind = pind

        def getNodeHash(self):
            for i, indices in enumerate(self.indexList):
                if i is 0:
                    h = indices
                else:
                    h += indices
            return h

        def getIndexList(self):
            return self.indexList

    def planning(self,):
        start_node = self.CoupledAstarNode(self.startIndices, 0.0, -1)
        goal_node = self.CoupledAstarNode(self.goalIndices, 0.0, -1)

        goal_id = goal_node.getNodeHash()
        open_set, closed_set = dict(), dict()
        open_set[start_node.getNodeHash()] = start_node

        cnt = 0
        while 1:
            cnt += 1
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(open_set[o], goal_node))
            curNode = open_set[c_id]

            if self.isGoal(curNode):
                print("Find goal")
                goal_node.pind = curNode.pind
                goal_node.cost = curNode.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = curNode

            # expand_grid search grid based on motion model
            curLocs = curNode.getIndexList()
            connLocs = [self.roadmap.get_connections(locId) for locId in curLocs]
            next_valid_sets = [s for s in itertools.product(*connLocs)]

            coordList = [self.roadmap.get_loc(loc_ind) for loc_ind in curLocs]
            print("Current Locs:", coordList)
            print("Iteration:", cnt)
            print("Cost:", curNode.cost)
            print()

            for indexList in next_valid_sets:
                # make new node
                distFromCurNode = self.sumOfDistancesBetweenIndexLists(curLocs, indexList)
                node = self.CoupledAstarNode(list(indexList), curNode.cost + distFromCurNode, c_id)
                n_id = node.getNodeHash()

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        return self.calcFinalPath(goal_node, closed_set)

    def calcFinalPath(self, goal_node, closed_set):
        locPath = [[] for i in range(self.robots.get_num_robots())]

        locs = goal_node.getIndexList()
        for i, loc in enumerate(locs):
            locPath[i].append(loc)

        pind = goal_node.pind
        while pind != -1:
            n = closed_set[pind]
            locs = n.getIndexList()
            for i, loc in enumerate(locs):
                locPath[i].append(loc)
            pind = n.pind
        for path in locPath:
            path.reverse()
        return locPath

    def calc_heuristic(self, curNode, goalNode):
        curIndexList = curNode.getIndexList()
        goalIndexList = goalNode.getIndexList()
        return self.sumOfDistancesBetweenIndexLists(curIndexList, goalIndexList)

    def plot_roadmap(self):
        print("Displaying Roadmap... May take time :)")
        edges = set()
        for i, _ in enumerate(self.roadmap.roadmap):
            for ii in range(len(self.roadmap.roadmap[i])):
                ind = self.roadmap.roadmap[i][ii]
                edge = (ind, i) if ind < i else (i, ind)
                if edge in edges:
                    continue
                else:
                    edges.add(edge)
                    plt.plot([self.roadmap.sample_locs[i][0], self.roadmap.sample_locs[ind][0]],
                             [self.roadmap.sample_locs[i][1], self.roadmap.sample_locs[ind][1]], "-k")
        plt.show(block=True)

    def verify_node(self, node):
        loc_list = node.getIndexList()
        # check if inside bounds
        for i, loc in enumerate(loc_list):
            # check for collisions
            for otherloc in loc_list[i+1:]:
                if loc == otherloc:
                    return False
        # check geometry
        coordList = [self.roadmap.get_loc(loc_ind) for loc_ind in loc_list]
        g = graph.Graph()
        g.initialize_from_location_list(coordList, self.robots.get_sensing_radius())
        eigval = g.get_nth_eigval(4)
        # if eigval < self.min_eigval:
        #     print("\nInvalid Configuration\n")
        #     return False

        return True

    def sumOfDistancesBetweenIndexLists(self, locationList, goalList):
        assert(len(locationList) == len(goalList))
        dists = []
        for i, _ in enumerate(locationList):
            dists.append(self.distanceBetweenLocs(locationList[i], goalList[i]))
        return sum(dists)

    def distanceBetweenLocs(self, loc1, loc2):
        x1, y1 = self.roadmap.get_loc(loc1)
        x2, y2 = self.roadmap.get_loc(loc2)
        delta_x = x1-x2
        delta_y = y1-y2
        return math.hypot(delta_x, delta_y)

    def isGoal(self, node):
        return (node.getIndexList() == self.goalIndices)
