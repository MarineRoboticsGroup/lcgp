#ifndef ROBOT_H_
#define ROBOT_H_


// #include <ros/ros.h>

#include "geom.h"
#include "plot.h"

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>

class Robot
{

	public:

		Robot(Point2d startLoc, int nId);
		Robot();
		~Robot();

		Point2d getCurrLoc() const { return loc; }
		int getRobotId() const { return id; }

		/**
		 * @brief      Returns the distance between robots and updates 'dists'
		 *             
		 *
		 * @param[in]  rob   Other robot to calculate distance from
		 *
		 */
		float distToRob(Robot rob);

		/**
		 * @brief      Moves the robot specified amounts in each direction. Has control saturation.
		 *
		 * @param[in]  x     The distance to move in x direction
		 * @param[in]  y     The distance to move in y direction
		 */
		void move(float x, float y);



	private:
		int id;
		float controlSat = .25;

		// current location
		Point2d loc;

		// 'i'th index is distance to robot 'i'
		// std::vector<float> ranges;

};


#endif