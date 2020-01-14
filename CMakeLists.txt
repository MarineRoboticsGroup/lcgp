cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)

# Set Project Name
project(RangeOnlyViz)

find_package(PythonLibs REQUIRED)
find_package(SCIP REQUIRED CONFIG)
find_package(gflags REQUIRED)
find_package(matplotlibcpp REQUIRED)

# This is really bad practice... see:
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/
# include_directories(${PYTHON_INCLUDE_DIRS}
# 	${PROJECT_SOURCE_DIR}/include
# 	${PROJECT_SOURCE_DIR}/include/matplotlib-cpp
# 	${SCIP_INCLUDE_DIRS}
# )

# This is really bad practice... see:
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/
# file(GLOB SOURCES
# 	${PROJECT_SOURCE_DIR}/src/*.cpp
# )
# Instead, create a library with all your code.
set(rov-src
	${PROJECT_SOURCE_DIR}/src/graph.cpp
	${PROJECT_SOURCE_DIR}/src/plot.cpp
	${PROJECT_SOURCE_DIR}/src/robot.cpp
	${PROJECT_SOURCE_DIR}/src/world.cpp
	)

add_library(rov-lib ${rov-src})

target_include_directories(rov-lib PUBLIC
	${PROJECT_SOURCE_DIR}/include
	${PYTHON_INCLUDE_DIRS}
	${SCIP_INCLUDE_DIRS}W
	)

target_link_libraries(rov-lib PUBLIC ${PYTHON_LIBRARIES} ${SCIP_LIBRARIES} matplotlibcpp)

# Add executable, and now one just needs to link it to your custom library.
add_executable(RangeOnlyViz src/main.cpp)
target_link_libraries(RangeOnlyViz rov-lib gflags)
