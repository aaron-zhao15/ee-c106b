execute_process(COMMAND "/home/cc/ee106b/sp23/class/ee106b-aas/ros_workspaces/proj1b/build/proj1_pkg/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/cc/ee106b/sp23/class/ee106b-aas/ros_workspaces/proj1b/build/proj1_pkg/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
