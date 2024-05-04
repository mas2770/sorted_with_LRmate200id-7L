#!/usr/bin/env python3

from __future__ import print_function
from six.moves import input
import numpy as np
import quaternion
import random
import sys
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pyrealsense2
import os
import time
from realsense_depth import *
import math
import statistics
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

class MoveGroupPythonInterface(object):

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "arm_group"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        group_name_gripper = "gripper_group"
        move_group_gripper = moveit_commander.MoveGroupCommander(group_name_gripper)
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=10,
        )
        
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()
        self.move_group_gripper = move_group_gripper
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
    
    def go_to_pose_goal(self, x_length,y_length, z_length=0.2, q1=0.0, q2=3.14/2, q3=0.0):
        move_group = self.move_group
        quaternion = tf.transformations.quaternion_from_euler(q1, q2, q3)  # Поворот на 90 градусов вокруг оси X
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
    def add_box(self, name,x,y,z,a1=0.05,a2=0.05,a3=0.05,q3=0.0,timeout=4):
        box_name = self.box_name
        scene = self.scene
        box_pose = geometry_msgs.msg.PoseStamped()
        
        box_pose.header.frame_id = "world" 
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0,q3)
        box_pose.pose.orientation.x = quaternion[0]
        box_pose.pose.orientation.y = quaternion[1]
        box_pose.pose.orientation.z = quaternion[2]
        box_pose.pose.orientation.w = quaternion[3]
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
        grasping_group = "gripper_group"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
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
    def open_close(self, name_fun):
        move_group_gripper = self.move_group_gripper
        move_group_gripper.set_named_target(name_fun)
        plan_success, plan, planning_time, error_code = move_group_gripper.plan()
        move_group_gripper.execute(plan, wait=True)

    def home(self):
        move_group = self.move_group
        move_group.set_named_target("home")
        plan_success, plan, planning_time, error_code = move_group.plan()
        move_group.execute(plan, wait=True)

def main():
    try:
        moveit = MoveGroupPythonInterface()
        dc = DepthCamera()

        cord_centre_cube = {"Blue": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Red": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Yellow": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Orange": [[0,0],[],[],[],0,[],0,[],[],0,0],
            "Purple": [[0,0],[],[],[],0,[],0,[],[],0,0]
        }
        colors = {
            "Blue": (255, 144, 30),
            "Red": (0, 0, 255),
            "Yellow": (0, 255, 255),
            "Orange": (0, 165, 255),
            "Purple": (204, 50, 153)
        }
        hsv_upper_and_lower = {
            "Blue": [[74, 100, 100], [130, 255, 255]],
            "Red": [[0, 120, 70], [10, 255, 255]],
            "Yellow": [[26, 50, 50], [35, 255, 255]],
            "Orange": [[11, 50, 50], [30, 255, 255]],
            "Purple": [[120, 50, 50], [170, 255, 255]]
        }

        for i in range(40):

            contours = []
            ret, depth_frame, color_frame = dc.get_frame()
            img = color_frame
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = cv2.blur(hsv, (5, 5))
            
            for key in colors:

                # if key != "Blue" or key != "Orange":
                #     continue
                lower = np.array(hsv_upper_and_lower[key][0])
                upper = np.array(hsv_upper_and_lower[key][1])
                mask = cv2.inRange(hsv, lower, upper)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=2)
                mask = cv2.dilate(mask, kernel, iterations=2)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours!=[] and contours is not None:
                    largest_contour = None
                    max_area = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            largest_contour = contour
                    
                    if cv2.contourArea(largest_contour) > 2500:
                        rect = cv2.minAreaRect(largest_contour)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cord_centre_cube[key][1].append(center_x)
                        cord_centre_cube[key][2].append(center_y)
                        cord_centre_cube[key][5].append(-rect[2])
                        cord_centre_cube[key][7].append(w)
                        cord_centre_cube[key][8].append(h)
                        cv2.circle(img,(center_x,center_y),5,(0,255,0),-1)
                        # try:
                        #     depth_centre = depth_frame[640,360]
                        #     depth_edge = depth_frame[640,260]
                        #     distance_between = math.sqrt((int(pow(depth_edge,2))-int(pow(depth_centre,2))))
                        #     transfer_coefficient = distance_between/100
                        #     if transfer_coefficient == 0.0:
                        #         continue
                        #     cord_centre_cube[key][3].append(transfer_coefficient)
                        #     # cord_centre_cube[key][4].append(depth_edge)
                        # except:
                        #     print('Невозможно определить глубину изображения!!!')
                        cv2.drawContours(img, [box], -1, colors[key], 7)
                        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow('img',img)
                        cv2.waitKey(1)
        cv2.destroyAllWindows()
        for key in cord_centre_cube:
            if key == "Blue":

                print(cord_centre_cube[key][1])
                print(cord_centre_cube[key][2])
                try:
                    # if key != 'Blue':
                    #     continue
                    cord_centre_cube[key][0][0] = statistics.mean(cord_centre_cube[key][1])
                    cord_centre_cube[key][0][1] = statistics.mean(cord_centre_cube[key][2])
                    cord_centre_cube[key][9] = statistics.mean(cord_centre_cube[key][7])
                    cord_centre_cube[key][10] = statistics.mean(cord_centre_cube[key][8])
                    # cord_centre_cube[key][4] = statistics.mean(cord_centre_cube[key][3])
                    cord_centre_cube[key][6] = statistics.mean(cord_centre_cube[key][5])
                    moveit.add_box(key,(cord_centre_cube[key][0][0])/1000,(cord_centre_cube[key][0][1])/1000,0.025,0.05,0.05,0.05,q3 = (cord_centre_cube[key][6]*3.14)/180)
                    # Построение и вывод графиков
                    # number_frame = []
                    # for i in range(1, len(cord_centre_cube[key][1])+1):
                    #     number_frame.append(i)
                    
                    # matplotlib.rcParams.update({'font.size': 12})
                    # plt.plot(number_frame, cord_centre_cube[key][1],'ob', label=r'$x$')
                    # plt.plot(number_frame, cord_centre_cube[key][2],'vr', label=r'$y$')
                    # plt.legend(fontsize=16)
                    # plt.xlabel('Номер фрейма')
                    # plt.ylabel('Координта центра по X и по Y')
                    # plt.title('График зависимости Y от X')
                    # plt.grid(which='major')
                    # plt.grid(which='minor', linestyle=':')
                    # plt.show()
                except:
                    print('Координаты центра кубика', key, 'не удалось получить')
        # moveit.add_box("wall_box_2",0.29,-0.3,0.11,0.01,0.3,0.22)
        # moveit.add_box("wall_box_3",0.44,-0.155,0.11,0.31,0.01,0.22)
        # moveit.add_box("wall_box_4",0.44,-0.445,0.11,0.31,0.01,0.22)
        # moveit.add_box("floor_box",0.45,-0.3,0.005,0.3,0.3,0.01)
        # moveit.add_box("barrier_main",0.6,0,0.25,0.6,0.02,0.5)
        # moveit.add_box("ceiling",0.0,0.0,0.9,2,2,0.02)
        # moveit.add_box("floor_main",0.0,0.0,0.0,2,2,0.001)
        # moveit.add_box("barrier_small",0.6,0.5,0.15,0.01,0.2,0.3)
        # moveit.add_box("barrier_big",0.22,0.4,0.075,0.15,0.01,0.13)
        
        # moveit.add_box(1,0.47,0.5,0.025, q3 = 3.14/3)
        # moveit.add_box(2,0.4,0.27,0.025, q3 = 3.14/6)
        # moveit.add_box(3,0.3,0.5,0.025)

        # moveit.open_close("open")

        # moveit.go_to_pose_goal(0.47,0.5,0.14, q1=((3.14/2) - (3.14/3)))
        # moveit.attach_box(1)
        # moveit.open_close("close")
        # moveit.go_to_pose_goal(0.47,0.5,0.2)
        # moveit.go_to_pose_goal(0.45,-0.3,0.35)
        # moveit.open_close("open")
        # moveit.detach_box(1)
        # moveit.remove_box(1)

        # moveit.go_to_pose_goal(0.4,0.27,0.14, q1=((3.14/2) - (3.14/6)))
        # moveit.attach_box(2)
        # moveit.open_close("close")
        # moveit.go_to_pose_goal(0.4,0.27,0.2, q1=((3.14/2) - (3.14/6)))
        # moveit.go_to_pose_goal(0.45,-0.3,0.35)
        # moveit.open_close("open")
        # moveit.detach_box(2)
        # moveit.remove_box(2)

        # moveit.go_to_pose_goal(0.3,0.5,0.14,q1=(3.14/2))
        # moveit.attach_box(3)
        # moveit.open_close("close")
        # moveit.go_to_pose_goal(0.3,0.5,0.2, q1=(3.14/2))
        # moveit.go_to_pose_goal(0.45,-0.3,0.35)
        # moveit.open_close("open")
        # moveit.detach_box(3)
        # moveit.remove_box(3)

        

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()
