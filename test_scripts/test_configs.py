from .. import math_utils
from .. import swarm
from .. import environment
from .. import plot

import time
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

colors = ['b','g','r','c','m','y']
nEig = 4

def tellme(phrase:str):
    """quick callback to embed phrases in the interactive plots

    Args:
        phrase (str): phrase to embed in plot
    """
    plt.title(phrase, fontsize=16)
    plt.draw()

def ClickPlaceNodes(sensing_radius:float=5, noise_model:str='add', noise_stddev:float=0.1):
    """interactive plot to generatively build networks by clicking to add nodes
    in locations

    Args:
        sensing_radius (float, optional): sensing radius of the swarm. Defaults to 5.
        noise_model (str, optional): the noise model being used (add or lognorm). Defaults to 'add'.
        noise_stddev (float, optional): standard deviation of the ranging sensor noise. Defaults to 0.1.
    """
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev)
    envBounds = (0, 1, 0, 1)
    env = environment.Environment(envBounds, setting='empty', num_obstacles=0)

    tellme('Click where you would like to place a node')
    plt.waitforbuttonpress()
    pts = []
    even = True
    while True:
        if even:
            tellme('Click for new node')
            temp = np.asarray(plt.ginput(1, timeout=-1))
            pts.append(tuple(temp[0]))
        else:
            tellme('Keypress to quit')
            if plt.waitforbuttonpress():
                    break
        even = not even
        robots.initialize_swarm_from_loc_list_of_tuples(pts)
        graph = robots.get_robot_graph()
        plot.plot(graph, env, blocking=True, animation=False)
        if robots.get_num_robots() >= 3:
            eigval = robots.get_nth_eigval(nEig)
            print(f"Eigval: {eigval}")
            plot.plot_nth_eigvec(robots,nEig)

def get_eigval_of_loc_list(loc_list:List[Tuple], sensing_radius:float=5, noise_model:str='add', noise_stddev:float=0.1):
    """prints out the least nontrivial eigenvalue based on the graph constructed
    from the location list

    Args:
        loc_list (List[Tuple]): [description]
        sensing_radius (float, optional): sensing radius of the swarm. Defaults to 5.
        noise_model (str, optional): the noise model being used (add or lognorm). Defaults to 'add'.
        noise_stddev (float, optional): standard deviation of the ranging sensor noise. Defaults to 0.1.

    Returns:
        float: the least nontrivial eigenvalue
    """
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)
    if robots.get_num_robots() >= 3:
        eigval = robots.get_nth_eigval(nEig)
    else:
        print("Needs more nodes")
    if False:
        min_x = min(loc_list[:,0]) - 1
        max_x = max(loc_list[:,0]) + 1
        min_y = min(loc_list[:,1]) - 1
        max_y = max(loc_list[:,1]) + 1
        env_bounds = (min_x, max_x, min_y, max_y)
        env = environment.Environment(env_bounds, setting='empty', num_obstacles=0)
        plot.plot(robots.get_robot_graph(), env, blocking=True, animation=False)
        plot.plot_nth_eigvec(robots,nEig)
        plt.show(block=True)

    return eigval

def generate_rotation_matrix(theta_degrees:float):
    """Generates a 2D rotation matrix 

    Args:
        theta_degrees (float): [the amount of the rotation, represented in degrees]

    Returns:
        [np.array]: [the rotation matrix representing the given rotation]
    """
    theta = (theta_degrees/180.) * np.pi
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
    return rotation_matrix

def test_configuration_from_loc_list(loc_list:List[Tuple], sensing_radius:float=5, noise_model:str='add', noise_stddev:float=0.1):
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)
    if robots.get_num_robots() >= 3:
        eigval = robots.get_nth_eigval(nEig)
        print(eigval)

    x_vals = [loc[0] for loc in loc_list]
    y_vals = [loc[1] for loc in loc_list]
    min_x = min(x_vals) - 1
    max_x = max(x_vals) + 1
    min_y = min(y_vals) - 1
    max_y = max(y_vals) + 1
    env_bounds = (min_x, max_x, min_y, max_y)
    env = environment.Environment(env_bounds, setting='empty', num_obstacles=0)
    plot.plot(robots.get_robot_graph(), env, blocking=False, animation=False, show_graph_edges=False)
    plot.plot_nth_eigvec(robots,nEig)
    plt.show(block=True)



if __name__ == '__main__':
    """This is a script to test out different network configurations
    for experimentation and validation of ideas
    """
    # ClickPlaceNodes()

    test_loc_list = [(0, 0), (1, 0), (.5, .5)]
    test_loc_list = [(0, 0), (1, 0), (1, 1)]
    test_loc_list = [(0, 0), (1, 0), (1, 1), (.6, .6)]
    test_configuration_from_loc_list(test_loc_list)



