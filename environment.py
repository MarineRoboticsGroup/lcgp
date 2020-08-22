import numpy as np
import math
import math_utils
import kdtree

class Environment:
    def __init__(self, bounds, numSquaresWide, numSquaresTall, setting, nObst):
        self.setting = setting
        self.obstacles = []
        self.bounds = bounds # xlb, xub, ylb, yub
        self.obstacleKDTree = None

        if setting == 'random':
            self.init_random_env(numObstacles=nObst)
        elif setting == 'curve_maze':
            self.init_curves_obstacles_env()
        elif setting == 'adversarial1':
            self.init_adversarial1_env()
        elif setting == 'adversarial2':
            self.init_adversarial2_env()
        elif not setting == 'empty':
            raise NotImplementedError

    ##### Modify and Initialize ############
    def add_obstacle(self, obs):
        self.obstacles.append(obs)
        return None

    def init_curves_obstacles_env(self):
        xlb, xub, ylb, yub = self.bounds

        radius = .75
        increments = 65
        # make left and right walls
        for y in np.linspace(ylb, yub, increments):
            cenLeft = (xlb, y)
            cenRight = (xub, y)
            obs = Obstacle(cenLeft, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenRight, radius)
            self.add_obstacle(obs)

        # make top and bottom walls
        for x in np.linspace(xlb, xub, increments):
            cenBot = (x, ylb)
            cenTop = (x, yub)
            obs = Obstacle(cenBot, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenTop, radius)
            self.add_obstacle(obs)

        # left divider
        span = yub-ylb
        dividerLen = 1/2
        xCen = xlb+(xub-xlb)/3
        for y in np.linspace(ylb, ylb + dividerLen*span, increments):
            cen = (xCen, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        # right divider
        xCen = xlb+2*(xub-xlb)/3
        for y in np.linspace(ylb+dividerLen*span, yub, increments):
            cen = (xCen, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_adversarial1_env(self):
        xlb, xub, ylb, yub = self.bounds
        radius = .75
        increments = 65
        # make left and right walls
        for y in np.linspace(ylb, yub, increments):
            cenLeft = (xlb, y)
            cenRight = (xub, y)
            obs = Obstacle(cenLeft, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenRight, radius)
            self.add_obstacle(obs)

        # make top and bottom walls
        for x in np.linspace(xlb, xub, increments):
            cenBot = (x, ylb)
            cenTop = (x, yub)
            obs = Obstacle(cenBot, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenTop, radius)
            self.add_obstacle(obs)

        # left divider
        span = yub-ylb
        dividerLen = 4/5
        xCen = xlb+(xub-xlb)/3
        for y in np.linspace(ylb, ylb + dividerLen*span, increments):
            cen = (xCen, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        # right divider
        xCen = xlb+2*(xub-xlb)/3
        for y in np.linspace(yub-dividerLen*span, yub, increments):
            cen = (xCen, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_adversarial2_env(self):
        xlb, xub, ylb, yub = self.bounds
        radius = .75
        increments = 65
        # make left and right walls
        for y in np.linspace(ylb, yub, increments):
            cenLeft = (xlb, y)
            cenRight = (xub, y)
            obs = Obstacle(cenLeft, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenRight, radius)
            self.add_obstacle(obs)

        # make top and bottom walls
        for x in np.linspace(xlb, xub, increments):
            cenBot = (x, ylb)
            cenTop = (x, yub)
            obs = Obstacle(cenBot, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenTop, radius)
            self.add_obstacle(obs)

        # left divider
        span = yub-ylb
        dividerLen = .45
        xCenLeft = xlb+(xub-xlb)/4
        for y in np.linspace(ylb, ylb + dividerLen*span, increments):
            cen = (xCenLeft, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        # right divider
        xCenRight = xub-(xub-xlb)/4
        for y in np.linspace(yub-dividerLen*span, yub, increments):
            cen = (xCenRight, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        # upper and lower dividers
        yCenUpper = yub-dividerLen*span
        yCenLower = ylb+dividerLen*span
        for x in np.linspace(xCenLeft, xCenRight, 20):
            cen = (x, yCenUpper)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)
            cen = (x, yCenLower)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_adversarial_easy_env(self):
        xlb, xub, ylb, yub = self.bounds
        radius = .75
        increments = 65
        # make left and right walls
        for y in np.linspace(ylb, yub, increments):
            cenLeft = (xlb, y)
            cenRight = (xub, y)
            obs = Obstacle(cenLeft, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenRight, radius)
            self.add_obstacle(obs)

        # make top and bottom walls
        for x in np.linspace(xlb, xub, increments):
            cenBot = (x, ylb)
            cenTop = (x, yub)
            obs = Obstacle(cenBot, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenTop, radius)
            self.add_obstacle(obs)

        # left divider
        span = yub-ylb
        dividerLen = 1/3
        xCenLeft = xlb+(xub-xlb)/3
        for y in np.linspace(ylb, ylb + dividerLen*span, increments):
            cen = (xCenLeft, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        # right divider
        xCenRight = xlb+2*(xub-xlb)/3
        for y in np.linspace(yub-dividerLen*span, yub, increments):
            cen = (xCenRight, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        # upper and lower dividers
        yCenUpper = yub-dividerLen*span
        yCenLower = ylb+dividerLen*span
        for x in np.linspace(xCenLeft, xCenRight, 20):
            cen = (x, yCenUpper)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)
            cen = (x, yCenLower)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_random_env(self, numObstacles=50):
        radius = 0.1
        for i in range(numObstacles):
            cnt = 0
            low = min(self.bounds)
            upp = max(self.bounds)
            radius = math_utils.genRandomTuple(lb=.5, ub=1, size=1)
            center = math_utils.genRandomTuple(lb=low, ub=upp, size=2)
            while not self.is_inside_bounds(center):
                center = math_utils.genRandomTuple(lb=low, ub=upp, size=2)
            obs = Obstacle(center, radius[0])
            self.add_obstacle(obs)

        if numObstacles > 0:
            self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    ###### Check Status ############
    def isFreeSpace(self, coords):
        if self.get_num_obstacles() == 0:
            return True
        if (not self.is_inside_bounds(coords)):
            return False
        idxs, dist = self.obstacleKDTree.search(np.array(coords).reshape(2, 1))
        if dist[0] <= self.obstacles[idxs[0]].get_radius():
            return False  # collision
        return True

    def isFreeSpaceLocListTuples(self, locList):
        for loc in locList:
            if not self.isFreeSpace(loc):
                return False
        return True

    def is_valid_path(self, startLoc, endLoc):
        if self.get_num_obstacles() == 0:
            return True
        if (not self.is_inside_bounds(startLoc)):
            return False
        if (not self.is_inside_bounds(endLoc)):
            return False
        sx, sy = startLoc
        gx, gy = endLoc
        move = [gx-sx, gy-sy]

        nsteps = 10
        for i in range(nsteps):
            dx = (i+1)/nsteps*move[0]
            dy = (i+1)/nsteps*move[1]
            loc = [sx+dx, sy+dy]
            idxs, dist = self.obstacleKDTree.search(np.array(loc).reshape(2, 1))
            if dist[0] <= self.obstacles[idxs[0]].get_radius():
                return False  # collision

        return True

    def is_inside_bounds(self, coords):
        x, y = coords
        xlb, xub, ylb, yub = self.bounds
        return ((xlb < x < xub) and (ylb < y < yub))

    ###### Accessors ############
    def get_obstacle_list(self,):
        return self.obstacles
        return None

    def get_obstacle_centers_list(self,):
        centers = []
        for obs in  self.obstacles:
            cen = list(obs.get_center())
            centers.append(cen)
        return centers

    def get_bounds(self,):
        return self.bounds

    def get_num_obstacles(self):
        return len(self.obstacles)
class Obstacle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_center(self,):
        return self.center
        return None

    def get_radius(self,):
        return self.radius
        return None

    def is_inside(self, coords):
        xpos, ypos = coords
        xcenter, ycenter = self.center
        delta = np.array([xpos-xcenter, ypos-ycenter])
        return (np.linalg.norm(delta, 2) < self.radius)

