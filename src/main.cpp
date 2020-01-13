#include "world.h"
#include "robot.h"
#include "geom.h"
#include <cmath>

#include <gflags/gflags.h>

DEFINE_double(noise, 0.0001, "this is the noise std dev param.");


void showRandomMovements(World w){
    for (int i = 0; i < 999; ++i)
    {

        // Plot all beacons
        w.plotRobots();

        w.fillRanges();
        // w.plotRangeCircles();
        w.plotRangeCircles(1);

        w.plotRobotConnections();
        w.setAxis(20, 20);
        // w.setAxisEqual();
        w.showMap("animation");
        w.randomMovements();

    }
}

void showStaticEnv(World w){
    // Plot all beacons
    w.plotRobots();

    w.fillRanges();
    // w.plotRangeCircles();
    // w.plotRangeCircles(0);
    w.plotRangeCircles(1);

    w.plotRobotConnections();
    w.setAxisEqual();
    w.showMap("static");
}

int main(int argc, char **argv) 
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    World w = World();

    // Add beacons to world
    w.addRobot(Point2d(1.0,1.5), FLAGS_noise);
    w.addRobot(Point2d(4.0,5.0), FLAGS_noise);
    w.addRobot(Point2d(2.0,7.0), FLAGS_noise);
    w.addRobot(Point2d(0.0,4.0), FLAGS_noise);
    // w.addBeacon(Point2d(1.0,9.0), FLAGS_noise);
    // w.addRobot(Point2d(3.7,2.0), FLAGS_noise);

    w.fillRanges();

    // w.printAdjGraph();
    // w.printGraphInfo();
    showStaticEnv(w);
    w.printGraphReal();
    // showRandomMovements(w);
       
    
    return 0;
}