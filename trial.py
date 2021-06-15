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
from planners import coupled_lazysp
from planners.prioritized_planning import prioritized_prm
from planners.prioritized_planning import prioritized_prm

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
    check_collision=False,
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
    :param      check_collision:      Whether to verify no inter-agent collisions occur
    :type       check_collision:      boolean
    """

    total_time = 0
    nonrigid_time = 0
    assert trajs is not None
    robot_size = 0.4

    traj_filepath = f"{cwd}/trajs/traj_{trial_timestamp}.txt"
    if not plan_name == "read_file":

        file_dir = os.path.dirname(traj_filepath)
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir, exist_ok=True)

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
        min_eigval = robots.get_nth_eigval(1)
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
            plot.plot(
                robots.get_robot_graph(),
                env,
                blocking=True,
                animation=False,
                goals=goals,
                clear_last=True,
                show_goals=True,
                show_graph_edges=True,
            )

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
            # Todo: move the collision checker further up the pipeline
            for otherRobotIndex in range(robotIndex + 1, robots.get_num_robots()):
                loc_1 = trajs[robotIndex][traj_indices[robotIndex]]
                loc_2 = trajs[otherRobotIndex][traj_indices[otherRobotIndex]]
                x_dif = abs(loc_1[0] - loc_2[0])
                y_dif = abs(loc_1[1] - loc_2[1])

                if (x_dif < robot_size) and (y_dif < robot_size):
                    currentIndex = max(
                        traj_indices[robotIndex], traj_indices[otherRobotIndex]
                    )
                    print(
                        "agents",
                        robotIndex,
                        "and",
                        otherRobotIndex,
                        "collide at index",
                        currentIndex,
                    )

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

    plot.test_trajectory_plot(
        robots.get_robot_graph(), env, goals, min_eigvals, robots.min_eigval
    )
    # plot.plot(
    #     robots.get_robot_graph(),
    #     env,
    #     blocking=False,
    #     animation=True,
    #     goals=goals,
    #     clear_last=True,
    #     show_goals=True,
    #     show_graph_edges=True,
    # )
    plt.pause(1)
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
        return False
    if not (env.is_free_space_loc_list_tuples(goals)):
        print("\nGoals Config Inside Obstacles")
        print()
        return False
    goalLoc = []
    for goal in goals:
        goalLoc += list(goal)

    # if priority planner we should check rigidity incrementally by iterating
    # over all of the robots
    if planner in priority_planners:
        for num_robots in range(3, swarm.get_num_robots() + 1):

            # check connected criteria for first 3 robots
            if num_robots < 4:

                # check that first 3 robots satisfy connected criteria at start
                start_locs = start_loc_list[:num_robots]
                for cur_robot_id, cur_robot_loc in enumerate(start_locs):

                    # check that is connected to one of previous locs
                    is_connected = False
                    for prev_robot_id, prev_robot_loc_id in enumerate(range(0, num_robots)):
                        prev_robot_loc = np.array(start_locs[prev_robot_loc_id])
                        dist_between = math_utils.calc_dist_between_locations(
                            cur_robot_loc, prev_robot_loc
                        )
                        print(f"Dist Between robot {cur_robot_id} and {prev_robot_id}: {dist_between}")
                        if dist_between < swarm.get_sensing_radius():
                            is_connected = True

                    if is_connected == False:
                        print("not connected at start")
                        return False

                # check that first 3 robots satisfy connected criteria at goal
                goal_locs = goals[:num_robots]
                for cur_robot_id, cur_robot_loc in enumerate(goal_locs):

                    # check that is connected to one of previous locs
                    is_connected = False
                    for prev_robot_id, prev_robot_loc_id in enumerate(range(0, num_robots)):
                        prev_robot_loc = goal_locs[prev_robot_loc_id]
                        dist_between = math_utils.calc_dist_between_locations(
                            cur_robot_loc, prev_robot_loc
                        )

                        print(f"Dist Between robot {cur_robot_id} and {prev_robot_id}: {dist_between}")
                        if dist_between < swarm.get_sensing_radius():
                            is_connected = True

                    if is_connected == False:
                        print("not connected at goal")
                        return False

            # check rigid criteria for all robots after first 3
            else:
                start_is_rigid = swarm.test_rigidity_from_loc_list(
                    start_loc_list[:num_robots]
                )
                goal_is_rigid = swarm.test_rigidity_from_loc_list(goals[:num_robots])
                if not start_is_rigid:
                    print(
                        f"\nStarting Config Insufficiently Rigid at Robot {num_robots-1}"
                    )
                    feasible = False
                if not goal_is_rigid:
                    print(f"\nGoal Config Insufficiently Rigid at Robot {num_robots-1}")
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
            return False
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
            return False

    # hasn't failed any checks so is a valid config and returning true
    return True


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

    rrt_planner = decoupled_rrt.DecoupledRRT(
        robot_graph=graph,
        goal_locs=goals,
        obstacle_list=obstacleList,
        bounds=environment.get_bounds(),
    )
    # robot_graph, goal_locs, obstacle_list, bounds,
    #              max_move_dist=3.0, goal_sample_rate=5, max_iter=500
    path = rrt_planner.planning()

    return path

def get_coupled_lazysp_path(robots, environment, goals):

    lazysp_planner = coupled_lazysp.LazySp(
        robots=robots, env=environment, goals=goals
    )

    traj = lazysp_planner.perform_planning()

    return traj



def get_priority_prm_path(robots, environment, goals, useTime):
    priority_prm = prioritized_prm.PriorityPrm(
        robots=robots, env=environment, goals=goals
    )
    traj = priority_prm.planning(useTime=useTime)
    return traj


def init_goals(
    swarmForm: str, setting: str, robots, bounds=None, shuffle_goals: bool = False
):

    if setting == "curve_maze":
        if robots.get_num_robots() == 20:
            goals = [
                (loc[0] + 24, loc[1] + 18) for loc in robots.get_position_list_tuples()
            ]
        else:
            goals = [
                (loc[0] + 24, loc[1] + 25) for loc in robots.get_position_list_tuples()
            ]
        if shuffle_goals:
            random.shuffle(goals)
        return goals
    elif setting == "random":
        assert bounds is not None, "need bounds for randomized setting"
        xub = bounds[1]
        xlb = xub - 15
        yub = bounds[3]
        ylb = xub - 15
        goals = [
            (np.random.uniform(xlb, xub), np.random.uniform(ylb, yub))
            for loc in robots.get_position_list_tuples()
        ]
        return goals
    elif swarmForm == "random":
        if setting == "curve_maze":
            goals = [
                (loc[0] + 18, loc[1] + 20) for loc in robots.get_position_list_tuples()
            ]
            if shuffle_goals:
                random.shuffle(goals)
            return goals
    elif swarmForm == "simple_vicon":
        if shuffle_goals:
            random.shuffle(goals)
        goals = [(loc[0] + 2, loc[1]) for loc in robots.get_position_list_tuples()]
        # goals = [(3.5, 0.9), (3.5, 1.5),
        #          (3.0, .3),
        #          (2.5, 0.9), (2.5, 1.5)] #difficult goals
        return goals

    elif swarmForm == "many_robot_simple_move_test":
        goals = [(loc[0] + 1, loc[1]) for loc in robots.get_position_list_tuples()]
        return goals
    elif swarmForm == "diff_end_times_test":
        goals = [
            (loc[0] + 5 + i, loc[1])
            for i, loc in enumerate(robots.get_position_list_tuples())
        ]
        return goals

    print(
        f"we do not have a predetermined set of goals for swarmForm: {swarmForm}, setting: {setting} "
    )
    raise NotImplementedError


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

    # if we are using certain configurations then we might generate goals or
    # starting conditions that aren't legal so we will cycle through
    # randomization until one is valid
    if swarmFormation == "random" or setting == "random":
        robots.initialize_swarm(
            env=env,
            bounds=bounds,
            formation=swarmFormation,
            nRobots=nRobots,
            min_eigval=min_eigval,
        )
        goals = init_goals(swarmFormation, setting, robots, bounds=envBounds)
        while not is_feasible_planning_problem(robots, env, goals, expName):
            robots.initialize_swarm(
                env=env,
                bounds=bounds,
                formation=swarmFormation,
                nRobots=nRobots,
                min_eigval=min_eigval,
            )
            goals = init_goals(
                swarmFormation, setting, robots, bounds=envBounds, shuffle_goals=True
            )
    else:
        robots.initialize_swarm(
            env=env,
            bounds=bounds,
            formation=swarmFormation,
            nRobots=nRobots,
            min_eigval=min_eigval,
        )

        goals = init_goals(swarmFormation, setting, robots)

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
    elif expName == "coupled_lazysp":
        trajs = get_coupled_lazysp_path(robots, env, goals)
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


def many_robot_simple_move_test():
    """Test function for planning for many robots but only moving one space to
    the right.
    """
    exp = "priority_prm"
    useTime = False
    useRelative = False

    # whether to show an animation of the planning
    showAnimation = False

    # whether to perform code profiling
    profile = False
    timestamp = None
    experimentInfo = (exp, useTime, useRelative, showAnimation, profile, timestamp)

    # the starting formation of the network
    swarmForm = "many_robot_simple_move_test"

    # the number of robots in the swarm
    nRobots = 40

    # the sensor noise model (additive or multiplicative gaussian)
    noise_model = "add"

    # the noise of the range sensors
    noise_stddev = 0.25

    # the sensing horizon of the range sensors
    sensingRadius = 6.5

    # the rigidity constraint on the network
    min_eigval = 0.0

    swarmInfo = (
        nRobots,
        swarmForm,
        sensingRadius,
        noise_model,
        min_eigval,
        noise_stddev,
    )

    # the layout of the environment to plan in
    setting = "empty"

    # the dimensions of the environment
    envSize = (35, 35)  # simulation

    # number of obstacles for random environment
    numObstacles = 0

    envInfo = (setting, envSize, numObstacles)

    seed = 99999999  # seed the randomization
    main(experimentInfo=experimentInfo, swarmInfo=swarmInfo, envInfo=envInfo, seed=seed)


def plan_anchor_only_test():
    """Test function for planning only the first 3 robots, which are assumed to
    all be anchor nodes (just enforces connectivity)
    """
    exp = "priority_prm"
    useTime = False
    useRelative = False

    # whether to show an animation of the planning
    showAnimation = False

    # whether to perform code profiling
    profile = False
    timestamp = None
    experimentInfo = (exp, useTime, useRelative, showAnimation, profile, timestamp)

    # the starting formation of the network
    swarmForm = "anchor_only_test"

    # the number of robots in the swarm
    nRobots = 3

    # the sensor noise model (additive or multiplicative gaussian)
    noise_model = "add"

    # the noise of the range sensors
    noise_stddev = 0.25

    # the sensing horizon of the range sensors
    sensingRadius = 100

    # the rigidity constraint on the network
    min_eigval = 0.0

    swarmInfo = (
        nRobots,
        swarmForm,
        sensingRadius,
        noise_model,
        min_eigval,
        noise_stddev,
    )

    # the layout of the environment to plan in
    setting = "curve_maze"

    # the dimensions of the environment
    envSize = (35, 35)  # simulation

    # number of obstacles for random environment
    numObstacles = 30

    envInfo = (setting, envSize, numObstacles)

    seed = 99999999  # seed the randomization
    main(experimentInfo=experimentInfo, swarmInfo=swarmInfo, envInfo=envInfo, seed=seed)


def different_end_times_test():
    """Test function for planning for many robots but only moving one space to
    the right.
    """
    exp = "priority_prm"
    useTime = False
    useRelative = False

    # whether to show an animation of the planning
    showAnimation = True

    # whether to perform code profiling
    profile = False
    timestamp = None
    experimentInfo = (exp, useTime, useRelative, showAnimation, profile, timestamp)

    # the starting formation of the network
    swarmForm = "diff_end_times_test"

    # the number of robots in the swarm
    nRobots = 20

    # the sensor noise model (additive or multiplicative gaussian)
    noise_model = "add"

    # the noise of the range sensors
    noise_stddev = 0.25

    # the sensing horizon of the range sensors
    sensingRadius = 6.5

    # the rigidity constraint on the network
    min_eigval = 0.01

    swarmInfo = (
        nRobots,
        swarmForm,
        sensingRadius,
        noise_model,
        min_eigval,
        noise_stddev,
    )

    # the layout of the environment to plan in
    setting = "empty"

    # the dimensions of the environment
    envSize = (35, 35)  # simulation

    # number of obstacles for random environment
    numObstacles = 0

    envInfo = (setting, envSize, numObstacles)

    seed = 99999999  # seed the randomization
    main(experimentInfo=experimentInfo, swarmInfo=swarmInfo, envInfo=envInfo, seed=seed)


if __name__ == "__main__":
    """
    This instantiates and calls everything.
    Any parameters that need to be changed should be accessible from here
    """
    run_tests = False

    if run_tests:
        print("Running simple test cases for planner")
        plan_anchor_only_test()
        many_robot_simple_move_test()
        different_end_times_test()

    else:
        # exp = "decoupled_rrt"
        exp = "priority_prm"
        # exp = "coupled_lazysp"
        # exp = "read_file"

        # whether to use time as extra planning dimension
        useTime = False

        # whether trajectory is recorded in relative moves or absolute positions
        useRelative = False

        # whether to show an animation of the planning
        showAnimation = False

        # whether to perform code profiling
        profile = True

        # the timestamp for replaying a recorded path (only when exp=="read_file")
        timestamp = 1600223009  # RRT
        # timestamp = 1600226369  # PRM
        # timestamp = 1

        experimentInfo = (exp, useTime, useRelative, showAnimation, profile, timestamp)

        # the starting formation of the network
        # swarmForm = 'square'
        # swarmForm = "test6"
        # swarmForm = "test8"
        swarmForm = "test20"
        # swarmForm = 'random'
        # swarmForm = "simple_vicon"

        # the number of robots in the swarm
        nRobots = 20

        # the sensor noise model (additive or multiplicative gaussian)
        noise_model = "add"

        # the noise of the range sensors
        noise_stddev = 0.25

        # the sensing horizon of the range sensors
        sensingRadius = 10

        # the rigidity constraint on the network
        min_eigval = 0.1

        swarmInfo = (
            nRobots,
            swarmForm,
            sensingRadius,
            noise_model,
            min_eigval,
            noise_stddev,
        )

        # the layout of the environment to plan in
        # setting = "random"
        setting = "curve_maze"
        # setting = 'adversarial1'
        # setting = 'adversarial2'
        # setting = 'simple_vicon'
        # setting = "obstacle_vicon"

        # the dimensions of the environment
        # envSize = (4.2, 2.4)  # vicon
        envSize = (35, 35)  # simulation

        # number of obstacles for random environment
        numObstacles = 10

        envInfo = (setting, envSize, numObstacles)

        seed = 99999999  # seed the randomization
        main(
            experimentInfo=experimentInfo,
            swarmInfo=swarmInfo,
            envInfo=envInfo,
            seed=301,
        )
