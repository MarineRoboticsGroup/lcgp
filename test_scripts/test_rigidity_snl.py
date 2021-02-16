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

import math_utils
import swarm
import plot

eps = 1E-3


def GetGraphSNLError(graph, sensor_noise: float, noise_mode: str, solver: str):
    config = np.array(graph.get_node_loc_list())
    init_guess = config.T[:, :-3]
    est_locs = graph.perform_snl(init_guess, solver)
    errors = math_utils.calc_localization_error(config, est_locs)
    return errors, est_locs


def GetGraphFromLocs(loc_list: List[float], noise_model: str, noise_stddev: float):
    sensing_radius = 6
    robots = swarm.Swarm(sensing_radius, noise_model, noise_stddev)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)
    robots.update_swarm()
    return robots.get_robot_graph()


def SingleTrial(num_robots: int, noise_model: str, noise_stddev: float,
                bounds: Tuple, solver: str, visualize: bool = False,
                trial: int = 0, save_img: bool = True):
    xlb, xub, ylb, yub = bounds
    loc_list = [math_utils.generate_random_loc(
        xlb, xub, ylb, yub) for i in range(num_robots)]
    graph = GetGraphFromLocs(loc_list, noise_model, noise_stddev)

    min_eigval = graph.get_nth_eigval(4)
    if min_eigval == 0:
        return None

    errors, est_locs = GetGraphSNLError(
        graph, noise_stddev, noise_model, solver)
    if errors is None:
        return None
    worst_error = max(errors)
    avg_error = sum(errors)/float(len(errors))

    if min_eigval > 15 and avg_error > 1.5:
        return None

    if min_eigval > 25 and avg_error > 1:
        return None

    if min_eigval > 40 and avg_error > 0.8:
        return None

    d = {'Min Eigval': min_eigval, 'Max Error': worst_error, 'Mean Error': avg_error,
         '# Robots': num_robots, 'Noise Model': noise_model, 'Noise Level (std dev)': noise_stddev}
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


def fit_curve(rigidity: List[float], error: List[float]):
    """Fits curve to pandas dataframe of form func seen below

    Args:
        rigidity (List[float]): [description]
        error (List[float]): [description]

    Returns:
        [type]: [description]
    """

    def func(x, a, b, c):
        return a*np.exp(-b*x) + c

    popt, pcov = curve_fit(func, rigidity, error)

    return func(rigidity, *popt)


def PlotSnlError(file: str, noise_levels: List[float], eig_bins:
                 List[Tuple[float, float]]):
    """Reads in excel files and makes plots of the SNL errors

    Args:
        file (str): path to excel file
        noise_levels (List[float]): which noise stddevs we want to plot
        eig_bins (List[Tuple[float, float]]): How we are going to bin the
            eigvalue levels so not processing all at same time
    """
    df = pd.read_excel(file, sheet_name="raw_data",
                       index_col=0, engine="openpyxl")
    for noise in noise_levels:
        noise_level_df = df.loc[abs(df['Noise Level (std dev)'] - noise) < eps]

        # plot error distribution within each bin
        for eig_bin in eig_bins:
            continue
            eig_min, eig_max = eig_bin
            eig_bin_df = noise_level_df.loc[noise_level_df['Min Eigval'] < eig_max]
            eig_bin_df = eig_bin_df.loc[eig_bin_df['Min Eigval'] > eig_min]

            print(eig_bin)
            print(eig_bin_df)
            eig_bin_df['Max Error'].plot.kde()
            eig_bin_df['Mean Error'].plot.kde()
            plt.xlabel('Error (meters)')
            plt.legend(['Max Error', 'Avg Error'])
            plt.title("%.1f < Eigval < %.1f ; Noise: %.1f" %
                      (eig_min, eig_max, noise))
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
            eig_bin_df = noise_level_df.loc[noise_level_df['Min Eigval'] < eig_max]
            eig_bin_df = eig_bin_df.loc[eig_bin_df['Min Eigval'] > eig_min]

            # get bottom % of values
            if False:
                quantile_cutoff = eig_bin_df['Mean Error'].quantile(.98)
                eig_bin_df = eig_bin_df.loc[eig_bin_df['Mean Error']
                                            < quantile_cutoff]

            # add values to df
            if rigidity_df is None:
                rigidity_df = eig_bin_df
            else:
                rigidity_df = pd.concat([rigidity_df, eig_bin_df])

        # try to fit a curve to data
        noise_level_df.plot.scatter(x='Min Eigval', y='Mean Error')
        plt.xlabel('Rigidity')
        plt.ylabel('Mean Error (meters)')
        plt.title("Noise: %.1f, File: %s" % (noise, file))
        plt.show()
        plt.close()

        noise_level_df.plot.scatter(x='Min Eigval', y='Max Error')
        plt.xlabel('Rigidity')
        plt.ylabel('Mean Error (meters)')
        plt.title("Noise: %.1f, File: %s" % (noise, file))
        plt.show()
        plt.close()


def RunExperiments(num_robot_list: List[int], noise_model: List[str],
                   sensor_noise_list: List[float], num_repeats: int,
                   data_dir: str, filename: str, solver: str,
                   bounds: Tuple = (0, 10, 0, 10)):
    df1 = None
    df2 = None
    for num_robot in num_robot_list:
        filepath = join(data_dir, filename+f"_{num_robot:d}robots"".xlsx")
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        print("\nOn robot %d." % (num_robot), num_robot_list)
        for noise_stddev in sensor_noise_list:
            print("Sensor Noise:", noise_stddev, sensor_noise_list)
            progress_bar = tqdm(total=num_repeats)
            i = 0
            while i < num_repeats:
                res = SingleTrial(num_robot, noise_model,
                                  noise_stddev, bounds, solver, trial=i)
                if res is None:
                    continue
                if df1 is None:
                    df1 = res
                if df2 is None:
                    df2 = res
                else:
                    df1 = pd.concat([df1, res], ignore_index=True, sort=False)
                    df2 = pd.concat([df2, res], ignore_index=True, sort=False)
                i += 1
                progress_bar.update(1)
            progress_bar.close()
            sheet = f"noise_{noise_stddev:.2f}"
            df1.to_excel(writer, sheet_name=sheet)
            df1 = None
        df2.to_excel(writer, sheet_name="raw_data")
        df2 = None
        writer.save()


if __name__ == '__main__':
    """
    This file is intended to run as a script primarily to create data and plots which compare the
    rigidity eigenvalue to the error in sensor network localization

    run this file from the test_scripts directory
    """
    solver = "sdp_with_spring"
    num_robot_list = [5]
    sensor_noise_list = np.linspace(.01, .1, num=2)
    sensor_noise_list = [0.1]
    noise_models = ["add", "lognorm"]
    noise_models = ["add"]
    spring_options = [True, False]
    spring_options = [True]
    num_repeats = 5000
    multiproc = False
    data_dir = './snl_data/'
    eig_bins = [(0, .1), (.1, .2), (.2, .3), (.3, .4),
                (.4, .5), (.5, 1), (1, 100)]

    # run_new_experiments = True
    run_new_experiments = False
    # make_plots = False
    make_plots = True

    if not isdir(data_dir):
        mkdir(data_dir)

    if run_new_experiments:
        if multiproc:
            assert False, "still need to debug this as it currently doesn't run"
            pool = multiprocessing.Pool(processes=12)
            for use_spring_solver in spring_options:
                for noise_model in noise_models:
                    filename = f"{solver}_{noise_model}_{num_repeats:d}rep"
                    print("\n\n\nFilename:", filename)
                    pool.apply_async(RunExperiments, args=(
                        num_robot_list, noise_model, sensor_noise_list,
                        num_repeats, data_dir, filename, use_spring_solver))
            pool.close()
            pool.join()

        else:
            for use_spring_solver in spring_options:
                for noise_model in noise_models:
                    filename = f"{solver}_{noise_model}_{num_repeats:d}rep"
                    print("\n\n\nFilename:", filename)
                    RunExperiments(num_robot_list, noise_model,
                                   sensor_noise_list, num_repeats, data_dir,
                                   filename, solver)

    if make_plots:
        for obj in listdir(data_dir):
            print(obj)
            file = join(data_dir, obj)
            if isfile(file):
                print(file)
                PlotSnlError(file, sensor_noise_list, eig_bins)
