cmake_minimum_required(VERSION 2.8.3)
project(motion_primitive_tree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

find_package(catkin REQUIRED COMPONENTS
		cpp_utils
		ros_utils
		collision_checker
		dynamic_reconfigure
		forward_arc_primitives
		roscpp
		trajectory_utils
)
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/CostFunctionWeights.cfg
)

catkin_package(
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIRS}
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
			cpp_utils
			ros_utils
			collision_checker
			dynamic_reconfigure
			forward_arc_primitives
			roscpp
			trajectory_utils
)

include_directories(include
      ${catkin_INCLUDE_DIRS}
)

link_directories(
      ${catkin_LIBRARY_DIRS}
)

add_definitions(
  ${catkin_DEFINITIONS}
)

add_library(${PROJECT_NAME}
  src/ListTree.cpp
  src/ListTreeBIAS.cpp
)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

add_dependencies(${PROJECT_NAME}
 	${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}
      ${catkin_LIBRARIES}
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
      add_definitions(-DARMA_NO_DEBUG)
endif()


install(TARGETS ${PROJECT_NAME}
      ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY cmake/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.cmake"
)
