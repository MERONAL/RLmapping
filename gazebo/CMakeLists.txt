project(amr_robots_gazebo)


find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
)

add_service_files(
    FILES Planning.srv
)

generate_messages(
    DEPENDENCIES 
    std_msgs
)