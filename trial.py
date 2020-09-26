import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import flamegraph
import time
import random
import subprocess
from typing import List, Tuple

import warnings

warnings.filterwarnings("ignore")

# custom libraries
import math_utils
import swarm
import environment
import plot

# planners
from planners import decoupled_rrt
from planners import coupled_astar
from planners import prioritized_prm

priority_planners = ["decoupled_rrt", "priority_prm"]
trial_timestamp = int(time.time())
cwd = os.getcwd()


def test_trajectory(
    robots,
    env,
    trajs,
    goals,
    plan_name,
    delay_animation=False,
    relativeTraj=False,
    sensor_noise=0.5,
):
    """
    Takes a generic input trajectory of absolute states
    and moves the swarm through the trajectory

    :param      robots:               The robots
    :type       robots:               Swarm object
    :param      env:                  The environment
    :type       env:                  Environment object
    :param      trajs:                The trajs
    :type       trajs:                list of lists of tuples of doubles
    :param      goals:                The goals
    :type       goals:                List of tuples
    :param      delay_animation:  Whether to delay the animation beginning
    :type       delay_animation:  boolean
    """

    total_time = 0
    nonrigid_time = 0
    assert trajs is not None

    traj_filepath = f"{cwd}/trajs/traj_{trial_timestamp}.txt"
    if not plan_name == "read_file":
        with open(traj_filepath, "w") as filehandle:
            for traj in trajs:
                filehandle.write("%s\n" % traj)

    traj_indices = [-1 for traj in trajs]
    final_traj_indices = [len(traj) - 1 for traj in trajs]
    move = []
    config = []
    min_eigvals = []
    mean_error_list = []
    only_plot_trajectories = False

    while not (traj_indices == final_traj_indices):
        min_eigval = robots.get_nth_eigval(4)
        min_eigvals.append(min_eigval)
        graph = robots.get_robot_graph()
        est_locs = graph.perform_snl()
        config = robots.get_position_list_tuples()
        error_list = math_utils.calc_localization_error(np.array(config), est_locs)
        mean_error = sum(error_list) / len(error_list)
        mean_error_list.append(mean_error)

        # Plotting options
        if only_plot_trajectories:
            plot.plot(
                robots.get_robot_graph(),
                env,
                blocking=False,
                animation=True,
                goals=goals,
                clear_last=True,
                show_goals=True,
                show_graph_edges=True,
            )
            # plot.plot(robots.get_robot_graph(), env, blocking=True, animation=False, goals=goals, clear_last=True, show_goals=True, show_graph_edges=True)

        else:
            plot.test_trajectory_plot(
                robots.get_robot_graph(), env, goals, min_eigvals, robots.min_eigval
            )

        trajectory_img_path = (
            f"{cwd}/figures/animations/traj_{trial_timestamp}_time{total_time}.png"
        )
        trajectory_img_path = f"{cwd}/figures/animations/image-{total_time}.png"
        plt.savefig(trajectory_img_path)

        total_time += 1
        move.clear()
        config.clear()
        for robotIndex in range(robots.get_num_robots()):
            # Increment trajectory for unfinished paths
            if traj_indices[robotIndex] != final_traj_indices[robotIndex]:
                traj_indices[robotIndex] += 1
            # Get next step on paths
            newLoc = trajs[robotIndex][traj_indices[robotIndex]]
            config.append(newLoc)
            move += list(newLoc)

        robots.move_swarm(move, is_relative_move=relativeTraj)
        robots.update_swarm()

        if min_eigval == 0 and False:
            print("Flexible Loc Est")
            print(est_locs)
            print()
            print("Gnd Truth Locs")
            print(np.array(config))
            print()

        if min_eigval < robots.min_eigval:
            nonrigid_time += 1
            print(f"{min_eigval} < {robots.min_eigval} at time {total_time}")
            # plot.plot_nth_eigvec(robots, 4)
            # plt.pause (5)
        if delay_animation and total_time == 1:
            plt.pause(10)

    plot.plot(
        robots.get_robot_graph(),
        env,
        blocking=False,
        animation=True,
        goals=goals,
        clear_last=True,
        show_goals=True,
        show_graph_edges=True,
    )
    trajectory_img_path = (
        f"{cwd}/figures/animations/traj_time{total_time}_{trial_timestamp}.png"
    )
    plt.savefig(trajectory_img_path)

    worst_error = max(mean_error_list)
    avg_error = sum(mean_error_list) / float(len(mean_error_list))
    print("Avg Localization Error:", avg_error)
    print("Max Localization Error:", worst_error)

    plt.close()

    plt.plot(min_eigvals)
    # plt.plot(loc_error)
    plt.hlines([robots.min_eigval], 0, len(min_eigvals))
    # plt.title("Minimum Eigenvalue over Time")
    plt.ylabel("Eigenvalue")
    plt.xlabel("time")
    plt.legend(["Eigvals", "Error"])
    plt.show()

    figure_path = (
        f"{cwd}/figures/plan_{plan_name}_noise_{sensor_noise}_{trial_timestamp}.png"
    )
    plt.savefig(figure_path)

    print("Total Time:", total_time)
    print("Bad Time:", nonrigid_time)


def is_feasible_planning_problem(swarm, env, goals: List, planner: str):
    feasible = True
    start_loc_list = swarm.get_position_list_tuples()
    graph = swarm.get_robot_graph()

    # show preliminary view of the planning problem
    # plot.plot(graph, env, show_graph_edges=True, blocking=True, goals=goals, animation=False, show_goals=True)

    if not (env.is_free_space_loc_list_tuples(swarm.get_position_list_tuples())):
        print("\nStart Config Inside Obstacles")
        print()
        feasible = False
    if not (env.is_free_space_loc_list_tuples(goals)):
        print("\nGoals Config Inside Obstacles")
        print()
        feasible = False
    goalLoc = []
    for goal in goals:
        goalLoc += list(goal)

    # plot.plot(graph, env, goals)

    # if priority planner we should check rigidity incrementally by iterating
    # over all of the robots
    if planner in priority_planners:
        for num_robot in range(3, swarm.get_num_robots()):
            start_is_rigid = swarm.test_rigidity_from_loc_list(
                start_loc_list[:num_robot]
            )
            goal_is_rigid = swarm.test_rigidity_from_loc_list(goals[:num_robot])
            if not start_is_rigid:
                print(f"\nStarting Config Insufficiently Rigid at Robot {num_robot}")
                feasible = False
            if not goal_is_rigid:
                print(f"\nGoal Config Insufficiently Rigid at Robot {num_robot}")
                feasible = False

    else:
        start_is_rigid = swarm.test_rigidity_from_loc_list(start_loc_list)
        goal_is_rigid = swarm.test_rigidity_from_loc_list(goals)
        if not start_is_rigid:
            print("\nStarting Config Insufficiently Rigid")
            print()
            graph = swarm.get_robot_graph()
            plot.plot_nth_eigvec(swarm, 4)
            plot.plot(
                graph, env, goals=goals, show_goals=True, blocking=True, animation=False
            )
            feasible = False
        if not goal_is_rigid:
            print("\nGoal Config Insufficiently Rigid")
            print()
            swarm.move_swarm(goalLoc, is_relative_move=False)
            swarm.update_swarm()
            graph = swarm.get_robot_graph()
            plot.plot_nth_eigvec(swarm, 4)
            plot.plot(
                graph, env, goals=goals, show_goals=True, blocking=True, animation=False
            )
            feasible = False
    return feasible


def read_traj_from_file(filename):
    trajs = []

    # open file and read the content in a list
    with open(filename, "r") as filehandle:
        for line in filehandle:
            traj = []

            line = line[:-1]
            # remove linebreak which is the last character of the string
            startInd = line.find("(")
            endInd = line.find(")")

            while startInd != -1:
                sect = line[startInd + 1 : endInd]
                commaInd = sect.find(",")
                x_coord = float(sect[:commaInd])
                y_coord = float(sect[commaInd + 1 :])
                coords = (x_coord, y_coord)
                traj.append(coords)

                line = line[endInd + 1 :]
                startInd = line.find("(")
                endInd = line.find(")")

            trajs.append(traj)
    return trajs


def convert_absolute_traj_to_relative(loc_lists):
    relMoves = [[(0, 0)] for i in loc_lists]

    for robotNum in range(len(loc_lists)):
        for i in range(len(loc_lists[robotNum]) - 1):
            x_old, y_old = loc_lists[robotNum][i]
            x_new, y_new = loc_lists[robotNum][i + 1]
            delta_x = x_new - x_old
            delta_y = y_new - y_old
            relMoves[robotNum].append((delta_x, delta_y))
    return relMoves


def make_sensitivity_plots_random_motions(robots, environment):
    """
    Makes sensitivity plots random motions.

    :param      robots:       The robots
    :type       robots:       { type_description }
    :param      environment:  The environment
    :type       environment:  { type_description }
    """
    vectorLength = 0.1

    for vectorLength in [0.15, 0.25, 0.5]:
        robots.initialize_swarm()

        predChanges = []
        actChanges = []
        predRatios = []

        predChangesCrit = []
        actChangesCrit = []
        predRatiosCrit = []

        for _ in range(500):

            origEigval = robots.get_nth_eigval(1)
            grad = robots.get_gradient_of_nth_eigval(1)
            if grad is False:
                # robots.show_swarm()
                break
            else:
                dirVector = math_utils.generate_random_vec(len(grad), vectorLength)
                predChange = np.dot(grad, dirVector)
                if origEigval < 1:
                    while predChange < vectorLength * 0.8:
                        dirVector = math_utils.generate_random_vec(
                            len(grad), vectorLength
                        )
                        predChange = np.dot(grad, dirVector)

                robots.move_swarm(dirVector)
                robots.update_swarm()
                newEigval = robots.get_nth_eigval(4)
                actChange = newEigval - origEigval

                predRatio = actChange / predChange

                if abs(predChange) > 1e-4 and abs(actChange) > 1e-4:
                    predChanges.append(predChange)
                    actChanges.append(actChange)
                    predRatios.append(predRatio)

                    if origEigval < 1:
                        predChangesCrit.append(predChange)
                        actChangesCrit.append(actChange)
                        predRatiosCrit.append(predRatio)

        if len(predChanges) > 0:
            # plot ratio
            print("Making plots for step size:", vectorLength, "\n\n")
            plt.figure()
            plt.plot(predRatios)
            plt.ylim(-3, 10)
            plt.show(block=False)
            title = "ratio of actual change to 1st order predicted change: {0}".format(
                (int)(vectorLength * 1000)
            )
            plt.title(title)
            rationame = "/home/alan/Desktop/research/ratio{0}.png".format(
                (int)(vectorLength * 1000)
            )
            plt.savefig(rationame)
            plt.close()

            # plot pred vs actual
            plt.figure()
            plt.plot(predChanges)
            plt.plot(actChanges)
            plt.show(block=False)
            title = "absolute change in eigenvalue: {0}".format(
                (int)(vectorLength * 1000)
            )
            plt.title(title)
            absname = "/home/alan/Desktop/research/abs{0}.png".format(
                (int)(vectorLength * 1000)
            )
            plt.savefig(absname)
            plt.close()


def get_decoupled_rrt_path(robots, environment, goals):
    obstacleList = environment.get_obstacle_list()
    graph = robots.get_robot_graph()

    rrt_planner = decoupled_rrt.RRT(
        robot_graph=graph,
        goal_locs=goals,
        obstacle_list=obstacleList,
        bounds=environment.get_bounds(),
    )
    # robot_graph, goal_locs, obstacle_list, bounds,
    #              max_move_dist=3.0, goal_sample_rate=5, max_iter=500
    path = rrt_planner.planning()
    return path


def get_coupled_astar_path(robots, environment, goals):
    a_star = coupled_astar.CoupledAstar(robots=robots, env=environment, goals=goals)
    traj = a_star.planning()
    return traj


def get_priority_prm_path(robots, environment, goals, useTime):
    priority_prm = prioritized_prm.PriorityPrm(
        robots=robots, env=environment, goals=goals
    )
    traj = priority_prm.planning(useTime=useTime)
    return traj


def init_goals(robots):
    # random config
    # goals = [(loc[0]+18, loc[1]+20) for loc in robots.get_position_list_tuples()]

    # curve environment
    goals = [(loc[0] + 24, loc[1] + 25) for loc in robots.get_position_list_tuples()]

    loc1 = (28, 19)
    loc2 = (31, 21)
    loc3 = (27.5, 21.5)
    loc4 = (32, 25)
    # goals = [loc1, loc2, loc3, loc4]
    loc5 = (27, 26)
    loc6 = (29.5, 27)
    loc7 = (27.5, 30)
    loc8 = (32.5, 29)
    # goals = [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8]

    # random.shuffle(goals)
    # print("Goals:", goals)
    return goals


def main(experimentInfo, swarmInfo, envInfo, seed=99999999):
    np.random.seed(seed)

    expName, useTime, useRelative, showAnimation, profile, timestamp = experimentInfo
    (
        nRobots,
        swarmFormation,
        sensingRadius,
        noise_model,
        min_eigval,
        noise_stddev,
    ) = swarmInfo
    setting, bounds, n_obstacles = envInfo

    envBounds = (0, bounds[0], 0, bounds[1])

    # Initialize Environment
    env = environment.Environment(envBounds, setting=setting, num_obstacles=n_obstacles)

    # Initialize Robots
    robots = swarm.Swarm(sensingRadius, noise_model, noise_stddev)
    if swarmFormation == "random":
        robots.initialize_swarm(
            env=env,
            bounds=bounds,
            formation=swarmFormation,
            nRobots=nRobots,
            min_eigval=min_eigval,
        )
        goals = init_goals(robots)
        while not is_feasible_planning_problem(robots, env, goals, expName):
            robots.initialize_swarm(
                env=env,
                bounds=bounds,
                formation=swarmFormation,
                nRobots=nRobots,
                min_eigval=min_eigval,
            )
            goals = init_goals(robots)
    else:
        robots.initialize_swarm(
            env=env, bounds=bounds, formation=swarmFormation, min_eigval=min_eigval
        )

    # Init goals
    goals = init_goals(robots)

    # Sanity checks
    assert is_feasible_planning_problem(robots, env, goals, expName)
    assert nRobots == robots.get_num_robots()

    ##### Perform Planning ######
    #############################
    startPlanning = time.time()
    if profile:
        fg_log_path = f"{cwd}/profiling/rgcp_flamegraph_profiling_{trial_timestamp}.log"
        fg_thread = flamegraph.start_profile_thread(fd=open(fg_log_path, "w"))

    if (
        expName == "decoupled_rrt"
    ):  # generate trajectories via naive fully decoupled rrt
        trajs = get_decoupled_rrt_path(robots, env, goals)
    elif expName == "coupled_astar":
        trajs = get_coupled_astar_path(robots, env, goals)
    elif expName == "priority_prm":
        trajs = get_priority_prm_path(robots, env, goals, useTime=useTime)
    elif expName == "read_file":
        assert (
            timestamp is not None
        ), "trying to read trajectory file but no timestamp specified"
        traj_filepath = f"{cwd}/trajs/traj_{timestamp}.txt"
        trajs = read_traj_from_file(traj_filepath)
    else:
        raise AssertionError

    endPlanning = time.time()
    print("Time Planning:", endPlanning - startPlanning)

    if profile:
        fg_thread.stop()
        fg_image_path = f"{cwd}/profiling/flamegraph_profile_{trial_timestamp}.svg"
        fg_script_path = f"{cwd}/flamegraph/flamegraph.pl"
        fg_bash_command = f"bash {cwd}/profiling/flamegraph.bash {fg_script_path} {fg_log_path} {fg_image_path}"
        subprocess.call(fg_bash_command.split(), stdout=subprocess.PIPE)

    if useRelative:
        print("Converting trajectory from absolute to relative")
        trajs = convert_absolute_traj_to_relative(trajs)

    if showAnimation:
        print("Showing trajectory animation")
        test_trajectory(
            robots,
            env,
            trajs,
            goals,
            expName,
            relativeTraj=useRelative,
            sensor_noise=noise_stddev,
        )


if __name__ == "__main__":
    """
    This instantiates and calls everything.
    Any parameters that need to be changed should be accessible from here
    """
    # exp = 'coupled_astar'
    # exp = 'decoupled_rrt'
    # exp = 'priority_prm'
    exp = "read_file"

    # timestamp = 1600223009  # RRT
    timestamp = 1600226369  # PRM
    # timestemp = None

    useTime = False
    useRelative = False
    showAnimation = True
    profile = False

    # swarmForm = 'square'
    # swarmForm = 'test6'
    swarmForm = "test8"
    # swarmForm = 'random'
    nRobots = 8
    noise_model = "add"
    sensingRadius = 6.5
    min_eigval = 0.25
    noise_stddev = 0.25

    # setting = 'random'
    setting = "curve_maze"
    # setting = 'adversarial1'
    # setting = 'adversarial2'
    envSize = (35, 35)
    numObstacles = 30

    experimentInfo = (exp, useTime, useRelative, showAnimation, profile, timestamp)
    swarmInfo = (
        nRobots,
        swarmForm,
        sensingRadius,
        noise_model,
        min_eigval,
        noise_stddev,
    )
    envInfo = (setting, envSize, numObstacles)

    main(experimentInfo=experimentInfo, swarmInfo=swarmInfo, envInfo=envInfo)
