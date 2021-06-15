import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# pylint: disable=import-error
import math_utils
import swarm
import environment
import plot

import time
import pandas as pd
import scipy.stats as st
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from typing import List, Tuple

colors = ["b", "g", "r", "c", "m", "y"]
nEig = 2


def tellme(phrase: str):
    """quick callback to embed phrases in the interactive plots

    Args:
        phrase (str): phrase to embed in plot
    """
    plt.title(phrase, fontsize=16)
    plt.draw()


def ClickPlaceNodes(
    sensing_radius: float = 5, noise_model: str = "add", noise_stddev: float = 0.1
):
    """interactive plot to generatively build networks by clicking to add nodes
    in locations

    Args:
        sensing_radius (float, optional): sensing radius of the swarm. Defaults to 5.
        noise_model (str, optional): the noise model being used (add or lognorm). Defaults to 'add'.
        noise_stddev (float, optional): standard deviation of the ranging sensor noise. Defaults to 0.1.
    """
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev)
    assert robots.get_num_robots() > 3, "not enough robots"
    assert nEig <= 2 * (
        robots.get_num_robots() - 3
    ), "should not have this many eigenvalues"
    envBounds = (0, 1, 0, 1)
    env = environment.Environment(envBounds, setting="empty", num_obstacles=0)

    tellme("Click where you would like to place a node")
    plt.waitforbuttonpress()
    pts = []
    even = True
    while True:
        if even:
            tellme("Click for new node")
            temp = np.asarray(plt.ginput(1, timeout=-1))
            pts.append(tuple(temp[0]))
        else:
            tellme("Keypress to quit")
            if plt.waitforbuttonpress():
                break
        even = not even
        robots.initialize_swarm_from_loc_list_of_tuples(pts)
        graph = robots.get_robot_graph()
        plot.plot(graph, env, blocking=True, animation=False)
        if robots.get_num_robots() >= 3:
            eigval = robots.get_nth_eigval(nEig)
            print(f"Eigval: {eigval}")
            plot.plot_nth_eigvec(robots, nEig)


def get_eigval_of_loc_list(
    loc_list: List[Tuple],
    sensing_radius: float = 5,
    noise_model: str = "add",
    noise_stddev: float = 0.1,
):
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

    assert robots.get_num_robots() > 3, "not enough robots"
    assert nEig <= 2 * (
        robots.get_num_robots() - 3
    ), "should not have this many eigenvalues"

    eigval = robots.get_nth_eigval(nEig)
    if False:
        min_x = min(loc_list[:, 0]) - 1
        max_x = max(loc_list[:, 0]) + 1
        min_y = min(loc_list[:, 1]) - 1
        max_y = max(loc_list[:, 1]) + 1
        env_bounds = (min_x, max_x, min_y, max_y)
        env = environment.Environment(env_bounds, setting="empty", num_obstacles=0)
        plot.plot(robots.get_robot_graph(), env, blocking=True, animation=False)
        plot.plot_nth_eigvec(robots, nEig)
        plt.show(block=True)

    return eigval


def generate_rotation_matrix(theta_degrees: float):
    """Generates a 2D rotation matrix

    Args:
        theta_degrees (float): [the amount of the rotation, represented in degrees]

    Returns:
        [np.array]: [the rotation matrix representing the given rotation]
    """
    theta = (theta_degrees / 180.0) * np.pi
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix


def get_fim_from_loc_list(
    loc_list: List[Tuple],
    sensing_radius: float = 999,
    noise_model: str = "add",
    noise_stddev: float = 0.1,
    num_anchors: int = 0,
):
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev, num_anchors)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)
    fim = robots.get_fisher_matrix(num_anchors=num_anchors)
    return fim


def test_configuration_from_loc_list(
    loc_list: List[Tuple],
    sensing_radius: float = 5,
    noise_model: str = "add",
    noise_stddev: float = 0.1,
    cost: str = "A",
):
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)

    assert robots.get_num_robots() > 3, "not enough robots"
    assert nEig <= 2 * (
        robots.get_num_robots() - 3
    ), "should not have this many eigenvalues"

    # x_vals = [loc[0] for loc in loc_list]
    # y_vals = [loc[1] for loc in loc_list]
    # min_x = min(x_vals) - 1
    # max_x = max(x_vals) + 1
    # min_y = min(y_vals) - 1
    # max_y = max(y_vals) + 1
    # env_bounds = (min_x, max_x, min_y, max_y)
    # env = environment.Environment(env_bounds, setting="empty", num_obstacles=0)
    # plot.plot(
    #     robots.get_robot_graph(),
    #     env,
    #     blocking=False,
    #     animation=False,
    #     show_graph_edges=False,
    # )
    # plot.plot_nth_eigvec(robots, nEig)
    # plt.show(block=True)

    # robots.print_all_eigvals()
    fim = robots.get_fisher_matrix(get_ungrounded=False)
    sub_fim = fim[:-2, :-2]
    # print(sub_fim)
    # print(f"Det FIM: {la.det(fim)}")
    # print(f"Det sub-FIM: {la.det(sub_fim)}")


def get_fim_cost(fim: np.ndarray, cost: str, num_anchors: int):
    assert isinstance(fim, np.ndarray)
    assert cost in ["A", "D", "E"]

    if cost == "A":
        val = math_utils.get_A_optimal_cost(fim)
    elif cost == "D":
        val = math_utils.get_D_optimal_cost(fim)
    elif cost == "E":
        # return math_utils.get_E_optimal_cost(fim)
        eigvals = math_utils.get_list_all_eigvals(fim)
        # print(eigvals)
        val = min(eigvals)
    else:
        raise NotImplementedError

    return val
    # return val / len(fim)


if __name__ == "__main__":
    """This is a script to test out different network configurations
    for experimentation and validation of ideas
    """
    # ClickPlaceNodes()
    import copy


    comb_cost_list = []
    max_cost_list = []
    min_cost_list = []
    comb_over_max = []
    comb_over_min = []
    cost_crit_list = []
    objective_list = []

    # orig_test_loc_list = [(0, 0), (1, 0), (1, 1), (2, 1)]

    cost_criterion = "E"
    n_anchors = 2
    num_orig_locs = 15
    num_trials = 999

    xub = 100
    yub = xub
    xlb = -xub
    ylb = xlb

    for _ in range(num_trials):
        orig_test_loc_list = [math_utils.generate_random_loc(xlb, xub, ylb, yub) for i in range(num_orig_locs)]

        # test list 1
        new_test_loc_list_1 = copy.deepcopy(orig_test_loc_list)
        rand_loc_1 = math_utils.generate_random_loc(xlb, xub, ylb, yub)
        new_test_loc_list_1.append(rand_loc_1)
        new_fim_1 = get_fim_from_loc_list(new_test_loc_list_1, num_anchors=n_anchors)
        cost1 = get_fim_cost(new_fim_1, cost=cost_criterion, num_anchors=n_anchors)

        # test list 2
        new_test_loc_list_2 = copy.deepcopy(orig_test_loc_list)
        rand_loc_2 = math_utils.generate_random_loc(xlb, xub, ylb, yub)
        new_test_loc_list_2.append(rand_loc_2)
        new_fim_2 = get_fim_from_loc_list(new_test_loc_list_2, num_anchors=n_anchors)
        cost2 = get_fim_cost(new_fim_2, cost=cost_criterion, num_anchors=n_anchors)

        min_cost = min(cost1, cost2)
        max_cost = max(cost1, cost2)

        # combined test list
        new_test_loc_list_comb = copy.deepcopy(orig_test_loc_list)
        new_test_loc_list_comb.append(rand_loc_1)
        new_test_loc_list_comb.append(rand_loc_2)
        new_fim_comb = get_fim_from_loc_list(
            new_test_loc_list_comb, num_anchors=n_anchors
        )
        cost_comb = get_fim_cost(
            new_fim_comb, cost=cost_criterion, num_anchors=n_anchors
        )
        # print(f"Cost 1: {cost1}")
        # print(f"Cost 2: {cost2}")
        # print(f"Min Cost: {min_cost}")
        # print(f"Max Cost: {max_cost}")
        # print(f"Combined Cost: {cost_comb}")

        # print(f"1 FIM Eigvals: {math_utils.get_list_all_eigvals(new_fim_1)}")
        # print(f"2 FIM Eigvals: {math_utils.get_list_all_eigvals(new_fim_2)}")
        # print(f"Combined FIM Eigvals: {math_utils.get_list_all_eigvals(new_fim_comb)}")
        # print()

        comb_cost_list.append(cost_comb)
        max_cost_list.append(max_cost)
        min_cost_list.append(min_cost)
        comb_over_max.append(cost_comb / max_cost)
        comb_over_min.append(cost_comb / min_cost)
        cost_crit_list.append(cost_criterion)

        # minimize A and maximize D, E
        if cost_criterion == "A":
            objective_list.append('minimize')
            # assert cost_comb <= max_cost + 1e-2
        else:
            objective_list.append('maximize')
            # assert cost_comb >= min_cost - 1e-2

    d = {
        "combined_cost": comb_cost_list,
        "max_cost": max_cost_list,
        "min_cost": min_cost_list,
        "combined_over_max_ratio": comb_over_max,
        "combined_over_min_ratio": comb_over_min,
        "cost_criterion": cost_crit_list,
        'objective': objective_list
    }
    cost_df = pd.DataFrame(data=d)
    cost_df = cost_df[cost_df['combined_over_min_ratio'] <= 3]
    print(f"Cost Criterion: {cost_criterion}, objective: {objective_list[0]}")
    print(f"Least Combined Over Min: {min(cost_df['combined_over_min_ratio'])}")
    print(f"Greatest Combined Over Max: {max(cost_df['combined_over_max_ratio'])}")
    # cost_df[['combined_over_min_ratio', 'combined_over_max_ratio']].hist(bins=50)
    cost_df[['combined_over_min_ratio']].hist(bins=100, density=True)

    # fit lognorm function
    res = cost_df[['combined_over_min_ratio']].to_numpy()
    shape, loc, scale = st.lognorm.fit(res, scale=.4)
    x = np.linspace(0,3, 200)
    lognorm_pdf = st.lognorm.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, lognorm_pdf, 'r-')
    plt.xlim(0, 3)
    plt.show(block=True)

    # for _ in range(99999):
    #     # copy list
    #     new_test_loc_list = copy.deepcopy(orig_test_loc_list)

    #     # add random loc to list
    #     cnt = 2
    #     for a in range(cnt):
    #         rand_loc = math_utils.generate_random_loc(-5, 5, -5, 5)
    #         new_test_loc_list.append(rand_loc)
    #         assert len(new_test_loc_list) > len(orig_test_loc_list)

    #     # get new fim submatrix
    #     new_fim = get_fim_from_loc_list(new_test_loc_list, num_anchors=n_anchors)
    #     new_sub_fim = new_fim[:-2*cnt, :-2*cnt]
    #     sub_fim_cost = get_fim_cost(new_sub_fim, cost=cost_criterion, num_anchors=n_anchors)
    #     print(f"Update Cost: {sub_fim_cost}, Orig Cost: {orig_fim_cost}")

    #     if cost_criterion == "A":
    #         assert sub_fim_cost <= orig_fim_cost
    #     else:
    #         assert sub_fim_cost >= orig_fim_cost
    # # test_loc_list = [(0, 0), (1, 0), (1, 1), (0.6, 0.6), (1.6, 0.6)]
    # # test_configuration_from_loc_list(test_loc_list)
