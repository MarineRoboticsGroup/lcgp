import graph

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from numpy import linalg as la
from typing import List, Tuple

colors = ["b", "g", "r", "c", "m", "y"]


def plot(
    graph,
    env,
    blocking: bool,
    animation: bool,
    goals: List[Tuple] = None,
    clear_last: bool = True,
    show_goals: bool = False,
    show_graph_edges: bool = True,
):
    """performs all plotting functionality

    Args:
        graph (Graph): [description]
        env (Environment): [description]
        blocking (bool): whether the call is meant to be blocking (must be True for animation)
        animation (bool): whether the call is meant to be for an animation
        goals (List[Tuple], optional): [description]. Defaults to None.
        clear_last (bool, optional): [description]. Defaults to True.
        show_goals (bool, optional): [description]. Defaults to False.
        show_graph_edges (bool, optional): [description]. Defaults to True.
    """
    if animation:
        assert not blocking

    if clear_last:
        clear_plot()
    if show_goals:
        plot_goals(goals)

    plot_graph(graph, show_graph_edges)

    plot_obstacles(env)
    set_x_lim(env.get_bounds()[0], env.get_bounds()[1])
    set_y_lim(env.get_bounds()[2], env.get_bounds()[3])

    plt.axis("off")
    plt.tick_params(
        axis="both",
        left="off",
        top="off",
        right="off",
        bottom="off",
        labelleft="off",
        labeltop="off",
        labelright="off",
        labelbottom="off",
    )
    if blocking:
        plt.show(block=True)
    else:
        if animation:
            plt.pause(0.1)
        plt.show(block=False)


def test_trajectory_plot(
    graph,
    env,
    goals: List[Tuple],
    min_eigvals: List,
    threshold_eigval: float,
    num_total_timesteps: int,
):
    assert isinstance(min_eigvals, list)
    assert isinstance(threshold_eigval, float)
    assert isinstance(num_total_timesteps, int)

    axs1 = plt.subplot(211)
    axs2 = plt.subplot(212)
    axs1.clear()
    axs2.clear()
    # set plot limits
    xlb, xub = (env.get_bounds()[0], env.get_bounds()[1])
    ylb, yub = (env.get_bounds()[2], env.get_bounds()[3])
    axs1.set_xlim(xlb, xub)
    axs1.set_ylim(ylb, yub)
    axs2.set_xlim(0, num_total_timesteps)
    axs2.set_ylim(-15, 15)

    # plot goals
    for i, goalLoc in enumerate(goals):
        axs1.scatter(goalLoc[0], goalLoc[1], color=colors[i % 6], marker="x")

    # plot graph
    node_locs = graph.get_node_loc_list()
    for i, node_loc in enumerate(node_locs):
        axs1.scatter(node_loc[0], node_loc[1], color=colors[i % 6])

    # plot edges
    edges = graph.get_graph_edge_list()
    nodeXLocs = [x[0] for x in node_locs]
    nodeYLocs = [x[1] for x in node_locs]
    for e in edges:
        xs = [nodeXLocs[e[0]], nodeXLocs[e[1]]]
        ys = [nodeYLocs[e[0]], nodeYLocs[e[1]]]
        axs1.plot(xs, ys, color="k")

    # plot obstacles
    obstacles = env.get_obstacle_list()
    for i, obs in enumerate(obstacles):
        if i%3 == 0:
            circ = plt.Circle(obs.get_center(), obs.get_radius(), color="r")
            axs1.add_artist(circ)

    # plot rigidity eigenvals
    timesteps = [i for i in range(len(min_eigvals))]
    _min_eigvals = np.array(min_eigvals)

    # interpolate to get finer resolution
    timestep_interp = np.linspace(0, max(timesteps), len(timesteps) * 20)
    _min_eigvals_interp = np.interp(timestep_interp, timesteps, _min_eigvals)

    # move data into desired format
    points = np.array([timestep_interp, _min_eigvals_interp]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # colormap and boundaries
    cmap = ListedColormap(["r", "g"])
    norm = BoundaryNorm([threshold_eigval], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(_min_eigvals_interp)
    lc.set_linewidth(2)

    # plot
    line = axs2.add_collection(lc)
    axs2.hlines([threshold_eigval], 0, num_total_timesteps)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 8)

    axs2.set_ylabel("Rigidity")

    # pause such that will play in 10s
    plt.pause(10.0/float(num_total_timesteps))
    plt.show(block=False)


def plot_trajectories(trajs, robots, env, goals):
    trajLen = [len(traj) for traj in trajs]
    maxLen = max(trajLen)
    for timestep in range(maxLen):
        for i, traj in enumerate(trajs):
            if traj == []:
                continue
            time = min(timestep, trajLen[i] - 1)
            loc = traj[time]
            plt.scatter(loc[0], loc[1], color=colors[i % 6])
            plt.plot(*zip(*traj), color=colors[i % 6])
        plot_obstacles(env)
        plot_goals(goals)
        if True:
            plt.pause(0.1)
            plt.show(block=False)
        else:
            plt.show(block=True)
        # if timestep == 0:
        #     plt.pause(5)
        clear_plot()
    plt.close()


def plot_graph(graph, show_graph_edges):
    node_locs = graph.get_node_loc_list()
    for i, node_loc in enumerate(node_locs):
        plt.scatter(node_loc[0], node_loc[1], color=colors[i % 6])

    if show_graph_edges:
        plot_edges(graph)


def plot_goals(goals):
    for i, goalLoc in enumerate(goals):
        plt.scatter(goalLoc[0], goalLoc[1], color=colors[i % 6], marker="x")


def plot_edges(graph):
    nodeLocations = graph.get_node_loc_list()
    edges = graph.get_graph_edge_list()
    nodeXLocs = [x[0] for x in nodeLocations]
    nodeYLocs = [x[1] for x in nodeLocations]
    for e in edges:
        xs = [nodeXLocs[e[0]], nodeXLocs[e[1]]]
        ys = [nodeYLocs[e[0]], nodeYLocs[e[1]]]
        plt.plot(xs, ys, color="k")


def plot_nth_eigvec(robots, n: int):
    eigpair = robots.get_nth_eigpair(n)
    _, eigvect = eigpair
    loc_list = robots.get_position_list_tuples()
    dir_vecs = np.array(
        [[eigvect[2 * i], eigvect[2 * i + 1]] for i in range(int(len(eigvect) / 2))]
    )
    dir_vecs = np.real(dir_vecs)
    dir_vecs = 0.5 * dir_vecs / la.norm(dir_vecs)
    for i, vector in enumerate(np.real(dir_vecs)):
        loc = loc_list[i]
        plt.arrow(loc[0], loc[1], vector[0], vector[1])


def plot_obstacles(env):
    obstacles = env.get_obstacle_list()
    fig = plt.gcf()
    ax = fig.gca()
    for obs in obstacles:
        circ = plt.Circle(obs.get_center(), obs.get_radius(), color="r")
        ax.add_artist(circ)


def set_x_lim(lb, ub):
    plt.xlim(lb, ub)


def set_y_lim(lb, ub):
    plt.ylim(lb, ub)


def clear_plot():
    plt.clf()
