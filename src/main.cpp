#include "world.h"
#include "robot.h"
#include "geom.h"
#include <cmath>


void showRandomMovements(World w){
    for (int i = 0; i < 999; ++i)
    {

        // Plot all robots
        w.plotRobots();

        w.fillRanges();
        // w.plotRangeCircles();
        // w.plotRangeCircles(1);

        w.plotRobotConnections();
        w.setAxis(20, 20);
        // w.setAxisEqual();
        w.showMap("animation");
        w.randomMovements();

    }
}

void showStaticEnv(World w){
    // Plot all robots
    w.plotRobots();

    w.fillRanges();
    // w.plotRangeCircles();
    w.plotRangeCircles(0);
    w.plotRangeCircles(1);

    w.plotRobotConnections();
    w.setAxisEqual();
    w.showMap("static");
}

int main() 
{
    // max number of robots
    int maxN = 100;
    World w = World(maxN);

    // Make robots and add to world
    Robot r0 = Robot(Point2d(1.0,1.5), 0, maxN);
    Robot r1 = Robot(Point2d(4.0,5.0), 1, maxN);
    Robot r2 = Robot(Point2d(2.0,5.0), 2, maxN);
    Robot r3 = Robot(Point2d(0.0,5.0), 3, maxN);
    Robot r4 = Robot(Point2d(3.7,5.0), 4, maxN);

    w.addRobot(r0);
    w.addRobot(r1);
    w.addRobot(r2);
    w.addRobot(r3);
    w.addRobot(r4);

    // showStaticEnv(w);
    // showRandomMovements(w);
    w.runGraphSample();
       
    
    return 0;
}