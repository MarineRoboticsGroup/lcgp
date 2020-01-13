cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)

# Set Project Name
project(RangeOnlyViz)

find_package(PythonLibs REQUIRED)
find_package(SCIP REQUIRED CONFIG)
find_package(gflags REQUIRED)

set(CUR_MODULE_PATH ${PROJECT_SOURCE_DIR}/conf)
set(CMAKE_MODULE_PATH ${CUR_MODULE_PATH} ${CMAKE_MODULE_PATH})

find_package(IPOPT REQUIRED)

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
	# In addition to the necessary python things for plotting.
	${PYTHON_INCLUDE_DIRS}
	${SCIP_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include/matplotlib-cpp
	${IPOPT_INCLUDE_DIRS})

target_link_libraries(rov-lib PUBLIC ${PYTHON_LIBRARIES} ${SCIP_LIBRARIES} ${IPOPT_LIBRARIES})

# Add executable, and now one just needs to link it to your custom library.
add_executable(RangeOnlyViz src/main.cpp)
target_link_libraries(RangeOnlyViz rov-lib gflags)
