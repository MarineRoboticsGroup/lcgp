import pandas as pd
import json

def display_results_from_file(results):
    # reformat
    for planner in results:
        #columns are tests, rows are values
        columns = []
        index = [
                "planning_time",
                "avg_loc_error",
                "max_loc_error",
                "avg_rmse_i",
                "max_rmse_i",
                "avg_rmse_t",
                "max_rmse_t",
                "avg_dist",
                ]
        data = [[] for i in range(8)]
        for test_case in results[planner]:
            columns.append(test_case)
            if results[planner][test_case]["success"]:
                for i in range(len(index)):
                    var = index[i]
                    if var in results[planner][test_case].keys():
                        data[i].append(results[planner][test_case][var])
                    else:
                        data[i].append("--")
            else:
                for i in range(len(index)):
                    data[i].append("--")


        #make a new file per planner
        df = pd.DataFrame(data, index=index, columns=columns)
        df_t = df.T
        json_formatted = df_t.to_json(orient='split')
        print(pd.read_json(json_formatted, orient='split'))
        # save

# load json
with open('./results_min_eig_1.json') as f:
    results = json.load(f)
display_results_from_file(results)
print("\n")

with open('./results_min_eig_2.json') as f:
    results = json.load(f)
display_results_from_file(results)
print("\n")

with open('./results_min_eig_3.json') as f:
    results = json.load(f)
display_results_from_file(results)