import sys

sys.path.insert(0, "./..")
import math_utils
import graph
import time


def generate_loc_list(num_locs: int, xlb: float, xub: float, ylb: float, yub: float):
    loc_list = []
    for i in range(num_locs):
        loc = math_utils.generate_random_loc(xlb, xub, ylb, yub)
        loc_list.append(loc)
    return loc_list


if __name__ == "__main__":
    xlb, xub, ylb, yub = (0, 5, 0, 5)
    num_locs = 50
    num_repeats = 100

    noise_model = "lognorm"
    noise_stddev = 0.4
    sensing_radius = 5

    loc_lists = [
        generate_loc_list(num_locs, xlb, xub, ylb, yub) for x in range(num_repeats)
    ]
    graph_list = []

    for i, loc_list in enumerate(loc_lists):
        test_graph = graph.Graph(noise_model, noise_stddev)
        test_graph.initialize_from_location_list(loc_list, sensing_radius)
        graph_list.append(test_graph)

    start = time.time()
    for graph in graph_list:
        graph.get_fisher_matrix()
    end = time.time()

    print(f"Made {num_repeats} FIMs in {end-start} seconds")