import numpy as np
import math
import math_utils
import kdtree


class Environment:
    def __init__(self, bounds, setting, num_obstacles):
        self.setting = setting
        self._obstacles = []
        self._bounds = bounds  # xlb, xub, ylb, yub
        self.obstacleKDTree = None

        if setting == 'random':
            self.init_random_env(numObstacles=num_obstacles)
        elif setting == 'curve_maze':
            self.init_curves_obstacles_env()
        elif setting == 'adversarial1':
            self.init_adversarial1_env()
        elif setting == 'adversarial2':
            self.init_adversarial2_env()
        elif setting == 'simple_vicon':
            self.init_simple_vicon_env()
        elif setting == 'obstacle_vicon':
            self.init_obstacle_vicon_env()
        elif setting == 'rectangle':
            self.init_rect_no_obstacle_env()
        elif not setting == 'empty':
            raise NotImplementedError

    """ Modify and Initialize """

    def add_obstacle(self, obs):
        self._obstacles.append(obs)
        return None

    def init_curves_obstacles_env(self):
        xlb, xub, ylb, yub = self._bounds

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
        xlb, xub, ylb, yub = self._bounds
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
        xlb, xub, ylb, yub = self._bounds
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
        xlb, xub, ylb, yub = self._bounds
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

    def init_simple_vicon_env(self):
        xlb, xub, ylb, yub = self._bounds
        radius = .1
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

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_obstacle_vicon_env(self):
        xlb, xub, ylb, yub = self._bounds
        radius = .2 #accounts for robot size
        increments = 40
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

        all_obstacles = [((3.0, 1.4), (.40, .50))]
                        #  ((3.0, .7), (.25, .25)),
                        #  ((1.0, 1.0), (.15, .15)),
                        #  ((3.0, 2.0), (.20, .30))] #((x,y), (depth, width))

        for i in range(len(all_obstacles)):
            (x_0, y_0) = all_obstacles[i][0]
            (block_depth, block_width) = all_obstacles[i][1]
            increments = max(int(max(block_depth, block_width)*10), 5)

            for y in np.linspace(y_0 - block_width/2, y_0 + block_width/2, increments):
                for x in np.linspace(x_0 - block_depth/2, x_0 + block_depth/2, increments):
                    cen = (x, y)
                    obs = Obstacle(cen, radius)
                    self.add_obstacle(obs)

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_random_env(self, numObstacles=50):
        radius = 0.1
        for _ in range(numObstacles):
            low = min(self._bounds)
            upp = max(self._bounds)
            radius = math_utils.generate_random_tuple(lb=.5, ub=1, size=1)
            center = math_utils.generate_random_tuple(lb=low, ub=upp, size=2)
            while not self.is_inside_bounds(center):
                center = math_utils.generate_random_tuple(
                    lb=low, ub=upp, size=2)
            obs = Obstacle(center, radius[0])
            self.add_obstacle(obs)

        if numObstacles > 0:
            self.obstacleKDTree = kdtree.KDTree(
                self.get_obstacle_centers_list())

    def init_rect_no_obstacle_env(self):
        xlb, xub, ylb, yub = self._bounds
        radius = .75
        # make left and right walls
        for y in np.linspace(ylb, yub, int(yub-ylb)*2):
            cenLeft = (xlb, y)
            cenRight = (xub, y)
            obs = Obstacle(cenLeft, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenRight, radius)
            self.add_obstacle(obs)

        # make top and bottom walls
        for x in np.linspace(xlb, xub, int(xub-xlb)*2):
            cenBot = (x, ylb)
            cenTop = (x, yub)
            obs = Obstacle(cenBot, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenTop, radius)
            self.add_obstacle(obs)

        self.obstacleKDTree = kdtree.KDTree(self.get_obstacle_centers_list())

    def init_rect_one_obstacle_env(self):
        xlb, xub, ylb, yub = self._bounds
        radius = .75
        # make left and right walls
        for y in np.linspace(ylb, yub, int(yub-ylb)*2):
            cenLeft = (xlb, y)
            cenRight = (xub, y)
            obs = Obstacle(cenLeft, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenRight, radius)
            self.add_obstacle(obs)

        # make top and bottom walls
        for x in np.linspace(xlb, xub, int(xub-xlb)*2):
            cenBot = (x, ylb)
            cenTop = (x, yub)
            obs = Obstacle(cenBot, radius)
            self.add_obstacle(obs)
            obs = Obstacle(cenTop, radius)
            self.add_obstacle(obs)

        # left divider
        span = yub-ylb
        dividerLen = 1/2
        xCen = xlb+(xub-xlb)/2
        for y in np.linspace(ylb, ylb + dividerLen*span, increments):
            cen = (xCen, y)
            obs = Obstacle(cen, radius)
            self.add_obstacle(obs)

    """ Check Status """

    def is_free_space(self, coords):
        if self.get_num_obstacles() == 0:
            return True
        if (not self.is_inside_bounds(coords)):
            return False
        indices, dist = self.obstacleKDTree.search(
            np.array(coords).reshape(2, 1))
        if dist[0] <= self._obstacles[indices[0]].get_radius():
            return False  # collision
        return True

    def is_free_space_loc_list_tuples(self, loc_list):
        for loc in loc_list:
            if not self.is_free_space(loc):
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

        num_steps = 10
        for i in range(num_steps):
            dx = (i+1)/num_steps*move[0]
            dy = (i+1)/num_steps*move[1]
            loc = [sx+dx, sy+dy]
            indices, dist = self.obstacleKDTree.search(
                np.array(loc).reshape(2, 1))
            if dist[0] <= self._obstacles[indices[0]].get_radius():
                return False  # collision

        return True

    def is_inside_bounds(self, coords):
        x, y = coords
        xlb, xub, ylb, yub = self._bounds
        return ((xlb < x < xub) and (ylb < y < yub))

    """ Accessors """

    def get_obstacle_list(self,):
        return self._obstacles

    def get_obstacle_centers_list(self,):
        centers = []
        for obs in self._obstacles:
            cen = list(obs.get_center())
            centers.append(cen)
        return centers

    def get_bounds(self,):
        return self._bounds

    def get_num_obstacles(self):
        return len(self._obstacles)


class Obstacle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_center(self,):
        return self.center

    def get_radius(self,):
        return self.radius

    def is_inside(self, coords):
        x_pos, y_pos = coords
        x_center, y_center = self.center
        dx = x_pos-x_center
        if dx > self.radius:
            return False
        dy = y_pos-y_center

        if dy > self.radius:
            return False

        delta = np.array([dx, dy])
        return (np.linalg.norm(delta, 2) < self.radius)
