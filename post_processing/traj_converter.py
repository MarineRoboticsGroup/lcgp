def write_grouped_by_robot(input_file: str, output_file: str):
    # load generated trajectories
    f1 = open(input_file, "r")
    all_trajs = f1.readlines()

    # reformat individual trajectories
    max_traj_length = 0
    output = []
    x_trajs = []
    y_trajs = []
    for traj in all_trajs:
        # get relevant data from input
        for char in "[()] ":
            traj = traj.replace(char, "")
        traj = list(traj.split(","))

        x_0 = float(traj[0])
        y_0 = float(traj[1])
        x_waypoints = []
        y_waypoints = []

        # convert
        for i in range(int(len(traj) / 2)):
            x = float(traj[2 * i]) - x_0
            y = float(traj[2 * i + 1]) - y_0
            x_waypoints.append(x)
            y_waypoints.append(y)

        x_trajs.append(x_waypoints)
        y_trajs.append(y_waypoints)
        output.extend([x_waypoints, "\n", y_waypoints, "\n\n"])
        # max_traj_length = max(max_traj_length, len(x_waypoints))

    # make sure each trajectory is the same length
    for i in range(len(output)):
        if type(output[i]) != list:  # filter out newlines
            continue
        traj = output[i]
        output[i] = traj + [traj[-1] for i in range(max_traj_length - len(traj))]

    traj_len = len(x_trajs)
    assert traj_len == len(y_trajs)
    with open(output_file, "w") as file:
        for robot_id in range(traj_len):
            file.write("Robot %d \n" % (robot_id+1))
            x_traj = str(x_trajs[robot_id])  # convert to string
            x_traj = x_traj[1:-1]  # get rid of brackets
            x_traj = x_traj.replace(",", "")  # trim commas
            file.write("[%s]\n" % str(x_traj)[1:-1])

            y_traj = str(y_trajs[robot_id])  # convert to string
            y_traj = y_traj[1:-1]  # get rid of brackets
            y_traj = y_traj.replace(",", "")  # trim commas
            file.write("[%s]" % (y_traj))
            file.write("\n\n")
            print("printed robot %d" % robot_id)

        # file.write("Waypoint Y Coordinates\n")
        # file.write("{")
        # for i, traj in enumerate(y_trajs):
        #     if i == traj_len-1:
        #     file.write("{%s} " % str(traj)[1:-1])
        # else:
        #         file.write("{%s}, " % str(traj)[1:-1])
        # file.write("}; \n\n")

    # saved newly formatted trajectories
    # output = [str(output[i]) for i in range(len(output))]
    # f2 = open(output_file, "w")
    # f2.writelines(output)
    # f2.close()


def write_grouped_by_coordinates(input_file: str, output_file: str):
    # load generated trajectories
    f1 = open(input_file, "r")
    all_trajs = f1.readlines()

    # reformat individual trajectories
    max_traj_length = 0
    output = []
    x_trajs = []
    y_trajs = []
    for traj in all_trajs:
        # get relevant data from input
        for char in "[()] ":
            traj = traj.replace(char, "")
        traj = list(traj.split(","))

        x_0 = float(traj[0])
        y_0 = float(traj[1])
        x_waypoints = []
        y_waypoints = []

        # convert
        for i in range(int(len(traj) / 2)):
            x = float(traj[2 * i]) - x_0
            y = float(traj[2 * i + 1]) - y_0
            x_waypoints.append(x)
            y_waypoints.append(y)

        output.extend([x_waypoints, "\n", y_waypoints, "\n\n"])
        max_traj_length = max(max_traj_length, len(x_waypoints))

        x_trajs.append(x_waypoints)
        y_trajs.append(y_waypoints)

    # make sure each trajectory is the same length
    for i in range(len(output)):
        if type(output[i]) != list:  # filter out newlines
            continue
        traj = output[i]
        output[i] = traj + [traj[-1] for i in range(max_traj_length - len(traj))]

    traj_len = len(x_trajs)
    assert traj_len == len(y_trajs)
    with open(output_file, "w") as file:
        file.write("Waypoint X Coordinates\n")
        file.write("{")
        for i, traj in enumerate(x_trajs):
            if i == traj_len - 1:
                file.write("{%s} " % str(traj)[1:-1])
            else:
                file.write("{%s}, " % str(traj)[1:-1])

        file.write("};\n\n")

        file.write("Waypoint Y Coordinates\n")
        file.write("{")
        for i, traj in enumerate(y_trajs):
            if i == traj_len - 1:
                file.write("{%s} " % str(traj)[1:-1])
            else:
                file.write("{%s}, " % str(traj)[1:-1])
        file.write("}; \n\n")

    # f2 = open(output_file, "w")
    # f2.writelines(x_trajs)
    # f2.writelines(y_trajs)
    # f2.close()


if __name__ == "__main__":
    #todo: support processing multiple files at once / take input file as an arg
    input_file = "traj_1004.txt"

    # grouping = "robots"  # write in groupings by robot trajectories
    # grouping = "coordinates" # write grouped by coordinates
    grouping = "both"

    if grouping == "robots":
        output_file = "robots_"+input_file
        print("writing grouped by robot ids")
        write_grouped_by_robot(input_file, output_file)

    elif grouping == "coordinates":
        output_file = "coords_"+input_file
        print("writing grouped by coords")
        write_grouped_by_coordinates(input_file, output_file)

    elif grouping == "both":
        output_file = "robots_"+input_file
        print("writing grouped by robot ids")
        write_grouped_by_robot(input_file, output_file)

        output_file = "coords_"+input_file
        print("writing grouped by coords")
        write_grouped_by_coordinates(input_file, output_file)

    else:
        raise NotImplementedError