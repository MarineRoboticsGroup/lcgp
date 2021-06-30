from os.path import join, isdir, isfile
from os import mkdir, listdir
from typing import List, Tuple
import matplotlib.pyplot as plt
from pandas.core.base import DataError
from pandas.core.frame import DataFrame
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
from scipy.optimize import curve_fit

import sys

sys.path.append("..")

# pylint: disable=import-error
import math_utils
import swarm
import plot

eps = 1e-3


def GetGraphSNLError(graph):
    config = np.array(graph.get_node_loc_list())
    current_guess = config[3:, :]

    # est_locs = graph.perform_snl(
    #     init_guess=current_guess.copy(), solver="spring_init_noise"
    # )
    est_locs = graph.perform_snl(init_guess=current_guess.copy(), solver="sp_optimize")
    if est_locs is None:
        return None, None
    # init_guess=current_guess.copy(), solver="spring_init_noise")
    errors = math_utils.calc_localization_error(config, est_locs)
    return errors, est_locs


def GetGraphFromLocs(loc_list: List[float], noise_stddev: float):
    sensing_radius = 6
    robots = swarm.Swarm(sensing_radius, "add", noise_stddev)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)
    robots.update_swarm()
    return robots.get_robot_graph()


def SingleTrial(
    num_robots: int,
    noise_stddev: float,
    bounds: Tuple,
    visualize: bool = False,
    trial: int = 0,
    save_img: bool = False,
):
    xlb, xub, ylb, yub = bounds
    loc_list = [
        math_utils.generate_random_loc(xlb, xub, ylb, yub) for i in range(num_robots)
    ]
    graph = GetGraphFromLocs(loc_list, noise_stddev)
    if not graph.nonanchors_are_k_connected(k=1):
        return None

    fim = graph.get_fisher_matrix()
    a_opt_val = math_utils.get_a_optimality(fim)
    e_opt_val = math_utils.get_e_optimality(fim)

    errors, est_locs = GetGraphSNLError(graph)
    worst_error = max(errors)
    avg_error = sum(errors) / float(len(errors))

    d = {
        "E-Opt Val": e_opt_val,
        "A-Opt Val": a_opt_val,
        "Max Error": worst_error,
        "Mean Error": avg_error,
        "# Robots": num_robots,
        "Noise Level (std dev)": noise_stddev,
    }
    res = pd.DataFrame(data=d, index=[0])

    # add visualization for trials to see how things are going
    if visualize:
        plot.plot_graph(graph, show_graph_edges=True)
        plot.plot_goals(est_locs)
        plt.show()

    if save_img:
        plot.plot_graph(graph, show_graph_edges=True)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(10, 6)
        image_dir = "./trial_images"
        if not isdir(image_dir):
            mkdir(image_dir)
        figure.savefig(f"{image_dir}/trial_{trial}.png", format="png")

    return res


def PlotSnlError(
    file: str, noise_levels: List[float], eig_bins: List[Tuple[float, float]]
):
    """Reads in excel files and makes plots of the SNL errors

    Args:
        file (str): path to excel file
        noise_levels (List[float]): which noise stddevs we want to plot
        eig_bins (List[Tuple[float, float]]): How we are going to bin the
            eigvalue levels so not processing all at same time
    """

    print("This function hasn't been integrated with the refactored code")
    raise NotImplementedError

    df = pd.read_excel(file, sheet_name="raw_data", index_col=0, engine="openpyxl")
    for noise in noise_levels:
        noise_level_df = df.loc[abs(df["Noise Level (std dev)"] - noise) < eps]

        # plot error distribution within each bin
        for eig_bin in eig_bins:
            continue
            eig_min, eig_max = eig_bin
            eig_bin_df = noise_level_df.loc[noise_level_df["Min Eigval"] < eig_max]
            eig_bin_df = eig_bin_df.loc[eig_bin_df["Min Eigval"] > eig_min]

            print(eig_bin)
            print(eig_bin_df)
            eig_bin_df["Max Error"].plot.kde()
            eig_bin_df["Mean Error"].plot.kde()
            plt.xlabel("Error (meters)")
            plt.legend(["Max Error", "Avg Error"])
            plt.title("%.1f < Eigval < %.1f ; Noise: %.1f" % (eig_min, eig_max, noise))
            plt.ylim((0, 0.7))
            plt.xlim(0, 20)
            plt.show()
            plt.close()

        # plot error vs rigidity
        rigidity_df = None
        for eig_bin in eig_bins:
            # continue
            # get values in eig bin
            eig_min, eig_max = eig_bin
            eig_bin_df = noise_level_df.loc[noise_level_df["Min Eigval"] < eig_max]
            eig_bin_df = eig_bin_df.loc[eig_bin_df["Min Eigval"] > eig_min]

            # get bottom % of values
            if False:
                quantile_cutoff = eig_bin_df["Mean Error"].quantile(0.98)
                eig_bin_df = eig_bin_df.loc[eig_bin_df["Mean Error"] < quantile_cutoff]

            # add values to df
            if rigidity_df is None:
                rigidity_df = eig_bin_df
            else:
                rigidity_df = pd.concat([rigidity_df, eig_bin_df])

        # try to fit a curve to data
        noise_level_df.plot.scatter(x="Min Eigval", y="Mean Error")
        plt.xlabel("Rigidity")
        plt.ylabel("Mean Error (meters)")
        plt.title("Noise: %.1f, File: %s" % (noise, file))
        plt.show()
        plt.close()

        noise_level_df.plot.scatter(x="Min Eigval", y="Max Error")
        plt.xlabel("Rigidity")
        plt.ylabel("Mean Error (meters)")
        plt.title("Noise: %.1f, File: %s" % (noise, file))
        plt.show()
        plt.close()


def RunExperiments(
    num_robot_list: List[int],
    sensor_noise: float,
    num_repeats: int,
    data_dir: str,
    filename: str,
    bounds: Tuple = (0, 10, 0, 10),
):
    df = None
    for num_robot in num_robot_list:
        filepath = join(data_dir, filename + f"_{num_robot:d}robots" ".xlsx")
        writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
        print("\nOn robot %d." % (num_robot), num_robot_list)
        noise_stddev = sensor_noise
        print("Sensor Noise:", noise_stddev)
        progress_bar = tqdm(total=num_repeats)
        i = 0
        while i < num_repeats:
            res = SingleTrial(num_robot, noise_stddev, bounds, trial=i)
            if res is None:
                continue
            if df is None:
                df = res
            else:
                df = pd.concat([df, res], ignore_index=True, sort=False)
            i += 1
            progress_bar.update(1)
        progress_bar.close()
        sheet = f"noise_{noise_stddev:.2f}"
        df.to_excel(writer, sheet_name=sheet)

        x = df["E-Opt Val"]
        y = df["Mean Error"]
        # y = df['Max Error']
        plt.scatter(x, y)
        # plt.yscale("log")

        # plot linear fit to data
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # print(p)
        # plt.plot(x, p(x), "r--")

        plt.xlim(0, 20)
        plt.ylim(0, 10)
        plt.show(block=True)

        writer.save()


if __name__ == "__main__":
    """
    This file is intended to run as a script primarily to create data and plots which compare the
    rigidity eigenvalue to the error in sensor network localization

    run this file from the test_scripts directory
    """
    num_robot_list = [9]
    sensor_noise = 0.1
    num_repeats = 99999
    multiproc = False
    data_dir = "./snl_data/"

    # run_new_experiments = True
    run_new_experiments = False
    # make_plots = False
    make_plots = True
    eig_bins = [
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 1),
        (1, 100),
    ]

    if not isdir(data_dir):
        mkdir(data_dir)

    filename = f"a_optimality_{sensor_noise}_{num_repeats:d}rep"
    print("\n\n\nFilename:", filename)
    RunExperiments(num_robot_list, sensor_noise, num_repeats, data_dir, filename)

    # if make_plots:
    #     for obj in listdir(data_dir):
    #         print(obj)
    #         file = join(data_dir, obj)
    #         if isfile(file):
    #             print(file)
    #             PlotSnlError(file, sensor_noise_list, eig_bins)

