import swarm
import math_utils
import plot

from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing

def GetGraphSNLError(graph, sensor_noise, noise_mode, use_spring_solver):
    config = np.array(graph.getNodeLocationList())
    init_guess = config.T[:, :-3]
    est_locs = graph.PerformSNL(sensor_noise, noise_mode, init_guess, use_spring_solver)
    errors = math_utils.CalculateLocalizationError(config, est_locs)
    return errors

def GetGraphFromLocs(loc_list, normalize_edge_len):
    robots = swarm.Swarm(6, normalize_edge_len)
    robots.initializeSwarmFromLocationListTuples(loc_list)
    robots.updateSwarm()
    return robots.getRobotGraph()

def SingleTrial(num_robots, sensor_noise, noise_mode, bounds, use_spring_solver, normalize_edge_len):
    xlb, xub, ylb, yub = bounds
    loc_list = [math_utils.genRandomLocation(xlb, xub, ylb, yub) for i in range(num_robots)]
    graph = GetGraphFromLocs(loc_list, normalize_edge_len)

    min_eigval = graph.getNthEigval(4)
    if min_eigval == 0:
        return None

    errors = GetGraphSNLError(graph, sensor_noise, noise_mode, use_spring_solver)
    if errors is None:
        return None
    worst_error = max(errors)
    avg_error = sum(errors)/float(len(errors))

    d = {'Min Eigval': min_eigval, 'Max Error': worst_error, 'Mean Error': avg_error, '# Robots': num_robots, 'Noise Mode': noise_mode, 'Noise Level (std dev)': sensor_noise}
    res = pd.DataFrame(data=d, index=[0])
    return res

def RunExperiments(num_robot_list, sensor_noise_list, noise_mode, num_repeats, filename, use_spring_solver, normalize_edge_len, bounds=(0,10,0,10)):
    df1 = None
    df2 = None
    for num_robot in num_robot_list:
        filepath = './snl_data/'+filename+f"_{num_robot:d}robots"".xlsx"
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        print("\nOn robot %d."%(num_robot), num_robot_list)
        for sensor_noise in sensor_noise_list:
            print("Sensor Noise:", sensor_noise, sensor_noise_list)
            pbar = tqdm(total=num_repeats)
            i = 0
            while i < num_repeats:
                res = SingleTrial(num_robot, sensor_noise, noise_mode, bounds, use_spring_solver, normalize_edge_len)
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
                pbar.update(1)
            pbar.close()
            sheet = f"noise_{sensor_noise:.2f}"
            df1.to_excel(writer, sheet_name=sheet)
            df1 = None
        df2.to_excel(writer, sheet_name="raw_data")
        df2 = None
        writer.save()



use_spring_solver = True
num_robot_list = [6, 10, 14]
sensor_noise_list = np.linspace(0, 1, num=6)
noise_mode = "relative"
normalize_mat_edges = True
num_repeats = 1000

# Note: absolute matrix is normalized by edge lengths

pool = multiprocessing.Pool(processes=4)
for use_spring_solver in [False]:
    for noise_mode in ["relative", "absolute"]:
        for normalize_mat_edges in [True, False]:
            if normalize_mat_edges:
                matrix_type = "absolute"
            else:
                matrix_type = "relative"

            if use_spring_solver:
                filename = "localsolve_"+noise_mode+f"noise_{num_repeats:d}rep_"+matrix_type+"matrix"
            else:
                filename = "sdponly_"+noise_mode+f"noise_{num_repeats:d}rep_"+matrix_type+"matrix"

            print("\n\n\nFilename:",filename)
            # RunExperiments(num_robot_list, sensor_noise_list, noise_mode, num_repeats, filename, use_spring_solver, normalize_mat_edges)
            pool.apply_async(RunExperiments, args = (num_robot_list, sensor_noise_list, noise_mode, num_repeats, filename, use_spring_solver, normalize_mat_edges))
pool.close()
pool.join()
