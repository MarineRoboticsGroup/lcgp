import graph

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import linalg as la

colors = ['b','g','r','c','m','y']


####### Single Frame Calls #######
def plot(graph, env, goals, animation=None, clear_last=True, show_goals=True, show_graph_edges=True):
    assert(animation is not None)

    if clear_last:
        clear_plot()
    if show_goals:
        plot_goals(goals)

    plot_graph(graph, show_graph_edges)

    plot_obstacles(env)
    plot_goals(goals)
    set_x_lim(env.get_bounds()[0], env.get_bounds()[1])
    set_y_lim(env.get_bounds()[2], env.get_bounds()[3])

    if animation:
        plt.pause(0.1)
        plt.show(block=False)
    else:
        plt.show(block=True)

def plot_trajectories(trajs, robots, env, goals, animation=True):
    trajLen = [len(traj) for traj in trajs]
    maxLen = max(trajLen)
    for timestep in range(maxLen):
        for i, traj in enumerate(trajs):
            if traj == []:
                continue
            time = min(timestep, trajLen[i]-1)
            loc = traj[time]
            plt.scatter(loc[0], loc[1], color=colors[i%6])
            plt.plot(*zip(*traj), color=colors[i%6])
        plot_obstacles(env)
        plot_goals(goals)
        if animation:
            plt.pause(0.1)
            plt.show(block=False)
        else:
            plt.show(block=True)
        # if timestep == 0:
        #     plt.pause(5)
        clear_plot()
    plt.close()

####### Atomic Calls #######
def plot_graph(graph, show_graph_edges):
    node_locs = graph.get_node_loc_list()
    for i, node_loc in enumerate(node_locs):
        plt.scatter(node_loc[0], node_loc[1], color=colors[i%6])
    
    if show_graph_edges:
        plot_edges(graph)

def plot_goals(goals):
    for i, goalLoc in enumerate(goals):
        plt.scatter(goalLoc[0], goalLoc[1], color=colors[i%6], marker='x')

def plot_edges(graph):
    nodeLocations = graph.get_node_loc_list()
    edges = graph.get_graph_edge_list()
    nodeXLocs = [x[0] for x in nodeLocations]
    nodeYLocs = [x[1] for x in nodeLocations]
    for e in edges:
        xs = [nodeXLocs[e[0]], nodeXLocs[e[1]]]
        ys = [nodeYLocs[e[0]], nodeYLocs[e[1]]]
        plt.plot(xs, ys, color='k')

def plot_nth_eigvec(robots, n):
    eigpair = robots.get_nth_eigpair(n)
    _, eigvect = eigpair
    robots.print_all_eigvalss()
    # K = robots.fisher_info_matrix
    # print(np.matmul(K, eigvect) - eigval*eigvect)
    # robots.print_fisher_matrix()
    # print("Eigval", eigval)
    loc_list = robots.get_position_list_tuples()
    dir_vecs = np.array([[eigvect[2*i], eigvect[2*i+1]] for i in range(int(len(eigvect)/2))])
    dir_vecs = np.real(dir_vecs)
    dir_vecs = 0.5*dir_vecs/ la.norm(dir_vecs)
    for i, vector in enumerate(np.real(dir_vecs)):
        loc = loc_list[i]
        plt.arrow(loc[0], loc[1],  vector[0], vector[1])

def plot_obstacles(env):
    obstacles = env.get_obstacle_list()
    fig = plt.gcf()
    ax = fig.gca()
    for obs in obstacles:
        circ = plt.Circle(obs.get_center(), obs.get_radius(), color='r')
        ax.add_artist(circ)

####### Basic Controls #######
def set_x_lim(lb, ub):
    plt.xlim(lb, ub)

def set_y_lim(lb, ub):
    plt.ylim(lb, ub)

def clear_plot():
    plt.clf()
