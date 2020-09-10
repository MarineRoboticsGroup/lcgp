from .. import swarm
from .. import math_utils

from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing

def GetGraphSNLError(graph, sensor_noise:float, noise_mode:str, solver:str):
    config = np.array(graph.get_node_loc_list())
    #pylint: disable=unsubscriptable-object
    init_guess = config.T[:, :-3]
    est_locs = graph.perform_snl(init_guess, solver)
    errors = math_utils.calc_localization_error(config, est_locs)
    return errors

def GetGraphFromLocs(loc_list:List[float], noise_model:str, noise_stddev:float):
    robots = swarm.Swarm(6, noise_model, noise_stddev)
    robots.initialize_swarm_from_loc_list_of_tuples(loc_list)
    robots.update_swarm()
    return robots.get_robot_graph()

def SingleTrial(num_robots:int, noise_model:str, noise_stddev:float, bounds:Tuple, solver:str):
    xlb, xub, ylb, yub = bounds
    loc_list = [math_utils.generate_random_loc(xlb, xub, ylb, yub) for i in range(num_robots)]
    graph = GetGraphFromLocs(loc_list, noise_model, noise_stddev)

    min_eigval = graph.get_nth_eigval(4)
    if min_eigval == 0:
        return None

    errors = GetGraphSNLError(graph, noise_stddev, noise_model, use_spring_solver)
    if errors is None:
        return None
    worst_error = max(errors)
    avg_error = sum(errors)/float(len(errors))

    d = {'Min Eigval': min_eigval, 'Max Error': worst_error, 'Mean Error': avg_error, '# Robots': num_robots, 'Noise Model': noise_model, 'Noise Level (std dev)': noise_stddev}
    res = pd.DataFrame(data=d, index=[0])
    return res

def RunExperiments(num_robot_list:List[int], noise_model:List[str], sensor_noise_list:List[float], num_repeats:int, filename:str, solver:str, bounds:Tuple=(0,10,0,10)):
    df1 = None
    df2 = None
    for num_robot in num_robot_list:
        filepath = './snl_data/'+filename+f"_{num_robot:d}robots"".xlsx"
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        print("\nOn robot %d."%(num_robot), num_robot_list)
        for noise_stddev in sensor_noise_list:
            print("Sensor Noise:", noise_stddev, sensor_noise_list)
            progress_bar = tqdm(total=num_repeats)
            i = 0
            while i < num_repeats:
                res = SingleTrial(num_robot, noise_model, noise_stddev, bounds, solver)
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
    """
    solver = "sdp_with_spring"
    num_robot_list = [6, 10, 14]
    sensor_noise_list = np.linspace(.1, .5, num=5)
    noise_model = "add"
    num_repeats = 1000

    # Note: absolute matrix is normalized by edge lengths

    pool = multiprocessing.Pool(processes=12)
    for use_spring_solver in [True, False]:
        for noise_model in ["add", "lognorm"]:
            filename = f"{solver}_{noise_model}_{num_repeats:d}rep"
            print("\n\n\nFilename:",filename)
            RunExperiments(num_robot_list, noise_model, sensor_noise_list, num_repeats, filename, solver)
    #         pool.apply_async(RunExperiments, args = (num_robot_list, noise_model, sensor_noise_list, num_repeats, filename, use_spring_solver))
    # pool.close()
    # pool.join()
