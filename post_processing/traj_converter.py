#load generated trajectories
f1 = open('input_trajs.txt', 'r')
all_trajs = f1.readlines()


#reformat individual trajectories
max_traj_length = 0
output = []
for traj in all_trajs:
    #get relevant data from input
    for char in "[()] ":
        traj = traj.replace(char, "")
    traj = list(traj.split(","))

    x_0 = float(traj[0])
    y_0 = float(traj[1])
    x_waypoints = []
    y_waypoints = []
    
    #convert
    for i in range(int(len(traj)/2)):
        x = float(traj[2*i]) - x_0
        y = float(traj[2*i+1]) - y_0
        x_waypoints.append(x)
        y_waypoints.append(y)

    output.extend([x_waypoints, "\n", y_waypoints, "\n\n"])
    max_traj_length = max(max_traj_length, len(x_waypoints))

#make sure each trajectory is the same length
for i in range(len(output)):
    if type(output[i]) != list: #filter out newlines
        continue
    traj = output[i]
    print(traj)
    print(traj[-1])
    print([traj[-1] for i in range(max_traj_length - len(traj))])
    output[i] = traj + [traj[-1] for i in range(max_traj_length - len(traj))]


#add list with as many zeros as waypoints
output.append([0 for i in range(len(output[0]))])


#saved newly formatted trajectories
output = [str(output[i]) for i in range(len(output))]
f2 = open('output_trajs.txt', 'w')
f2.writelines(output)
f2.close()