#include "world.h"
#include "robot.h"
#include "geom.h"
#include <cmath>




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

int main() 
{
    World w = World();

    // Add beacons to world
    w.addRobot(Point2d(1.0,1.5));
    w.addRobot(Point2d(4.0,5.0));
    w.addRobot(Point2d(2.0,5.0));
    w.addRobot(Point2d(0.0,5.0));
    w.addBeacon(Point2d(1.0,5.0));
    // w.addRobot(Point2d(3.7,5.0));
    w.printAdjGraph();
    showStaticEnv(w);
    // showRandomMovements(w);
       
    
    return 0;
}