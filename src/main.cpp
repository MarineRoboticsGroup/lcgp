#include "world.h"
#include "robot.h"
#include "geom.h"
#include <cmath>

#include <gflags/gflags.h>

DEFINE_double(noise, 0.0001, "this is the noise std dev param.");
DEFINE_bool(eq_axis, true, "flag to axes equal (ideal for plotting circles).");
DEFINE_bool(show_all_ranges, false, "flag to plot all range circles.");
DEFINE_bool(show_est_rob_range, false, "flag to plot range circles based on estimated robot location.");
DEFINE_int32(show_rob_range, -1, "param to plot range circles to specific robot.");
DEFINE_bool(show_rob_con, false, "flag to plot all connections between robots.");
DEFINE_bool(show_rob, true, "flag to plot robots.");
DEFINE_bool(show_est_rob, true, "flag to plot estimated robot location.");
DEFINE_bool(random_move, false, "flag to plot random robot movement animation.");


void showRandomMovements(World w){
    for (int i = 0; i < 999; ++i)
    {

    // Plot true robot locations
        if (FLAGS_show_rob)
        {
            w.plotRobots();
        }

    // plot estimated locations
        if (FLAGS_show_est_rob)
        {
            w.plotEstLocs();
        }

    // Plot range circles
        if (FLAGS_show_all_ranges)
        {
            w.plotRangeCircles();
        }
        else if (FLAGS_show_est_rob_range)
        {
            w.plotRangeCirclesEst();
        }
        else if (FLAGS_show_rob_range >= 0)
        {
            w.plotRangeCircles(FLAGS_show_rob_range);
        }

    // plot connections between robots
        if (FLAGS_show_rob_con)
        {
            w.plotRobotConnections();
        }

        if (FLAGS_eq_axis)
        {
            w.setAxisEqual();
        } 
        else
        {
            w.adjustLims();
            w.setAxis();
        }

        w.showMap("animation");
        w.randomMovements();
        w.fillRanges();
        w.realizeGraph();

    }
}

void showRobotsStatic(World w){
    // Plot true robot locations
    if (FLAGS_show_rob)
    {
        w.plotRobots();
    }

    // plot estimated locations
    if (FLAGS_show_est_rob)
    {
        w.plotEstLocs();
    }

    // Plot range circles
    if (FLAGS_show_all_ranges)
    {
        w.plotRangeCircles();
    }
    else if (FLAGS_show_est_rob_range)
    {
        w.plotRangeCirclesEst();
    }
    else if (FLAGS_show_rob_range >= 0)
    {
        w.plotRangeCircles(FLAGS_show_rob_range);
    }

    // plot connections between robots
    if (FLAGS_show_rob_con)
    {
        w.plotRobotConnections();
    }

    if (FLAGS_eq_axis)
    {
        w.setAxisEqual();
    } 
    else
    {
        w.adjustLims();
        w.setAxis();
    }
    w.showMap("static");
}


int main(int argc, char **argv) 
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    World w = World();

    // Add beacons to world
    w.addRobot(Point2d(0.0,0.0), FLAGS_noise);
    w.addRobot(Point2d(5.0,0.0), FLAGS_noise);
    w.addRobot(Point2d(10.0,0.0), FLAGS_noise);
    w.addRobot(Point2d(0.0,5.0), FLAGS_noise);
    // w.addBeacon(Point2d(1.0,9.0), FLAGS_noise);
    // w.addRobot(Point2d(3.7,2.0), FLAGS_noise);

    w.fillRanges();
    w.realizeGraph();

    if (FLAGS_random_move)
    {
        showRandomMovements(w);
    }
    else 
    {
        showRobotsStatic(w);   
    }
    // w.printAdjGraph();
    // w.printGraphInfo();
    
    return 0;
}