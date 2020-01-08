cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)

# Set Project Name
project(RangeOnlyViz)

find_package(PythonLibs REQUIRED)
find_package(SCIP REQUIRED CONFIG)

include_directories(${PYTHON_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include/matplotlib-cpp
	${SCIP_INCLUDE_DIRS}
)

file(GLOB SOURCES
	${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Add executable
add_executable(RangeOnlyViz ${SOURCES})
# add_executable(RangeOnlyViz src/main.cpp)

target_link_libraries(RangeOnlyViz ${PYTHON_LIBRARIES} ${SCIP_LIBRARIES})