#ifndef ROBOT_H_
#define ROBOT_H_


// #include <ros/ros.h>

#include "geom.h"
#include "plot.h"

#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>

class Robot
{

	public:

		Robot(Point2d startLoc, int nId, bool isRob);
		Robot(Point2d startLoc, int nId, bool isRob, float stdDev);
		Robot();
		~Robot();

		Point2d getCurrLoc() const { return loc; }
		int getId() const { return id; }
		float getStdDev() const { return rangeStddev; }
		float getControlSat() const { return controlSat; }

		/**
		 * @brief      Returns the distance between beacons and updates 'dists'
		 *             
		 *
		 * @param[in]  rob   Other robot to calculate distance from
		 *
		 */
		float distTo(Robot rob);

		/**
		 * @brief      Moves the robot specified amounts in each direction. Has
		 *             control saturation. Will only move if isRobot == true
		 *
		 * @param[in]  x     The distance to move in x direction
		 * @param[in]  y     The distance to move in y direction
		 */
		void move(float x, float y);



	private:
		bool isRobot = true;
		int id;
		float controlSat = .25;
		float rangeStddev = .01;

		// current location
		Point2d loc;

};


#endif