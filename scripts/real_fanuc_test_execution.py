#!/usr/bin/env python3

from __future__ import print_function
from six.moves import input
import numpy as np
import quaternion
import random
import sys
import actionlib
import tf
import copy
import rospy
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float64MultiArray
import std_msgs
import franka_gripper.msg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

try:
    from math import pi, tau, dist, fabs, cos
except:  
    from math import pi, fabs, cos, sqrt
    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def all_close(goal, actual, tolerance):
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)
    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        d = dist((x1, y1, z1), (x0, y0, z0))
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)
    return True

class MoveGroupPythonInterfaceTutorial(object):

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        # group_name_gripper = "gripper_group"
        # move_group_gripper = moveit_commander.MoveGroupCommander(group_name_gripper)
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=10,
        )
        
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        print(eef_link)
        group_names = robot.get_group_names()
        # self.move_group_gripper = move_group_gripper
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
    
    def go_to_pose_goal(self, x_length,y_length, z_length=0.2, orientation =[0.0, 3.14/2, 0.0]):
        move_group = self.move_group
        quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])  # Поворот на 90 градусов вокруг оси X
        pose_goal = geometry_msgs.msg.Pose()  
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]
        pose_goal.position.x = x_length  
        pose_goal.position.y = y_length
        pose_goal.position.z = z_length
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        box_name = self.box_name
        scene = self.scene
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0
            is_known = box_name in scene.get_known_object_names()
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True
            rospy.sleep(0.1)
            seconds = rospy.get_time()
        return False
    def add_box(self, name,x,y,z,a1=0.05,a2=0.05,a3=0.05,timeout=4):
        box_name = self.box_name
        scene = self.scene
        box_pose = geometry_msgs.msg.PoseStamped()
        
        box_pose.header.frame_id = "base_link" 
        box_pose.pose.orientation.w = 1.0
        box_name = str(name)
        box_pose.pose.position.x = x
        box_pose.pose.position.y = y
        box_pose.pose.position.z = z
        scene.add_box(box_name, box_pose, size=(a1, a2, a3))
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_box(self, name,timeout=4):
        box_name = str(name)
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        group_names = self.group_names
        grasping_group = "manipulator"
        # touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=['tool0'])
        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout
        )

    def detach_box(self,name1, timeout=4):
        box_name = str(name1)
        scene = self.scene
        eef_link = self.eef_link
        scene.remove_attached_object(eef_link, name=box_name)
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )

    def remove_box(self, name, timeout=4):
        box_name = self.box_name
        scene = self.scene
        scene.remove_world_object(str(name))
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )

    def home(self):
        move_group = self.move_group
        move_group.set_named_target("home")
        plan_success, plan, planning_time, error_code = move_group.plan()
        move_group.execute(plan, wait=True)

def main():
    try:
        tutorial = MoveGroupPythonInterfaceTutorial()

        # tutorial.add_box("wall_1",0.6,-0.3,0.11,0.01,0.3,0.22)
        # tutorial.add_box("wall_2",0.29,-0.3,0.11,0.01,0.3,0.22)
        # tutorial.add_box("wall_3",0.44,-0.155,0.11,0.31,0.01,0.22)
        # tutorial.add_box("wall_4",0.44,-0.445,0.11,0.31,0.01,0.22)
        # tutorial.add_box("floor",0.45,-0.3,0.005,0.3,0.3,0.01)
        tutorial.add_box("wall_main_1",0.6,0,0.25,0.6,0.02,0.5)
        tutorial.add_box("floor_main_1",0.0,0.0,1.1,2,2,0.001)
        tutorial.add_box("wall_main_2",-0.3,0,0.55,0.001,2,1.1)
        tutorial.add_box("floor_main_2",0.0,0.0,-0.01,2,2,0.001)
        # tutorial.add_box("wall_5",0.6,0.5,0.15,0.01,0.2,0.3)
        # tutorial.add_box("wall_6",0.22,0.4,0.075,0.15,0.01,0.13)
        # tutorial.add_box(1,0.3,0.3,0.025)
        # tutorial.add_box(2,0.5,0.2,0.025)
        # tutorial.add_box(3,0.35,0.5,0.025)
        tutorial.go_to_pose_goal(0.5,0.5,0.15, orientation =[0, 3.14, 0])
        tutorial.go_to_pose_goal(0.5,-0.5,0.15, orientation =[0, 3.14, 3.14/2])
    
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()
