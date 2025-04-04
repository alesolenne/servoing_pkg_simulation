#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from std_srvs.srv import SetBool
from fake_servo.kinematics import *
from fake_servo.tools import *
from pytransform3d.rotations import quaternion_from_matrix, axis_angle_from_quaternion

# Constants
JOINT_NAMES_ARM = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]
JOINT_NAMES_WRIST = [
    'qbmove2_motor_1_joint', 'qbmove2_motor_2_joint', 'qbmove2_shaft_joint',
    'qbmove2_deflection_virtual_joint', 'qbmove2_stiffness_preset_virtual_joint'
]
STIFFNESS_MAX = 1.0
ERROR_TRANSLATION_THRESHOLD = 0.005
ERROR_ORIENTATION_THRESHOLD = 0.01

def call_service(service_name, service_type, request):
    """Call a ROS service and handle exceptions."""
    rospy.wait_for_service(service_name)
    try:
        client = rospy.ServiceProxy(service_name, service_type)
        response = client(request)
        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to {service_name} failed: {e}")
        return None

def check_q_goal(q_goal, use_wrist):
    """Check if the robot and wrist have reached the desired position."""
    while True:
        joint_states = rospy.wait_for_message('/joint_states', JointState)

        if use_wrist:
            wrist_states = rospy.wait_for_message('/robot_arm/gripper/qbmove2/joint_states', JointState)

        if joint_states.name == JOINT_NAMES_ARM and (not use_wrist or wrist_states.name == JOINT_NAMES_WRIST):
            q_arm_actual = np.array(joint_states.position).reshape((7, 1))
            if use_wrist:
                q_wrist_actual = np.array(wrist_states.position[2]).reshape((1, 1))
                q_actual = np.vstack((q_arm_actual, q_wrist_actual))
                e_q = q_goal - q_actual

            else:
                q_actual = q_arm_actual
                e_q = q_goal - q_actual

            if np.linalg.norm(e_q, 2) < 0.01:
                return q_actual

def initialize_state(use_wrist):
    """Initialize the robot and wrist state."""
    while True:
        joint_states = rospy.wait_for_message('/joint_states', JointState)
        if use_wrist:
            wrist_states = rospy.wait_for_message('/robot_arm/gripper/qbmove2/joint_states', JointState)

        if joint_states.name == JOINT_NAMES_ARM and (not use_wrist or wrist_states.name == JOINT_NAMES_WRIST):
            q_arm = np.array(joint_states.position).reshape((7, 1))
            if use_wrist:
                q_wrist = np.array(wrist_states.position[2]).reshape((1, 1))
                q = np.vstack((q_arm, q_wrist))
                return q
            else:
                return q_arm

def feedback(trans_0B, rot_0B, trans_0V, rot_0V, q, use_wrist):
    """PBVS Visual servoing control law.

    Parameters
    ----------
    trans_ij : [3x1] np.ndarray
               translation vector of j with respect to i
    
    rot_ij : [3x3] np.ndarray
             rotation matrix from i to j
            
    Returns
    -------
    dq : [7x1] np.ndarray
         PBVS control law joints velocity

    e: [6x1] np.ndarray
       Position and orientation error vector
    """

    # Compute the Jacobian and the transformation matrix from base to end-effector
    (J, T_0E) = get_jacobian(q, use_wrist)
    T_0B = hom_matrix(trans_0B, rot_0B)
    T_0V = hom_matrix(trans_0V, rot_0V)

    # Extract translation and rotation components from the transformation matrices
    t_0V, t_0B = T_0V[:3, 3], T_0B[:3, 3]
    R_0V, R_0B = T_0V[:3, :3], T_0B[:3, :3]

    # Compute the relative rotation matrix and convert it to quaternion and axis-angle representation
    R_e = np.matmul(R_0B, np.transpose(R_0V))
    q_e = quaternion_from_matrix(R_e)
    q_e_axisangle = axis_angle_from_quaternion(q_e)
    r, theta = q_e_axisangle[:3], q_e_axisangle[3]

    # Compute the translation and orientation error vectors
    e_t = t_0B - t_0V
    e_o = np.sin(theta) * r
    e = np.vstack((e_t, e_o)).reshape((6, 1))

    # Construct the interaction matrix for orientation control
    I, Z = np.identity(3), np.zeros((3, 3))
    L_e = -0.5 * sum(np.matmul(skew_matrix(R_0B[:, i]), skew_matrix(R_0V[:, i])) for i in range(3))
    L = np.block([[I, Z], [Z, np.linalg.pinv(L_e)]])

    # Compute the pseudo-inverse of the Jacobian
    J_pinv = np.linalg.pinv(J)

    # Define proportional gain matrices for position and orientation control
    K_p, K_o = 1 * np.identity(3), 1 * np.identity(3)
    K = np.block([[K_p, Z], [Z, K_o]])

    # Compute the joint velocity command using the PBVS control law
    q_dot = np.linalg.multi_dot([J_pinv, L, K, e])

    # Return the computed joint velocities and the error vector
    return q_dot, e

def servoing(use_wrist):
    """Perform visual servoing."""
    pub_arm = rospy.Publisher(
        "/position_joint_trajectory_controller/command",
        JointTrajectory, queue_size=10
    )
    if use_wrist:
        pub_wrist = rospy.Publisher(
            "/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command",
            JointTrajectory, queue_size=10
        )

    q = initialize_state(use_wrist)
    x = 0.0

    while True:
        try:
            (t_0B, q_0B) = listener.lookupTransform('/panda_link0', '/object', rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

    prev_time = rospy.get_time()
    rospy.sleep(1.0)
    while not rospy.is_shutdown():
        try:
            (t_0V, q_0V) = listener.lookupTransform('/panda_link0', '/tool_extremity', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        t_0B, q_0B = np.array(t_0B), np.array(q_0B)
        t_0V, q_0V = np.array(t_0V), np.array(q_0V)



        q_dot, e = feedback(t_0B, q_0B, t_0V, q_0V, q, use_wrist)
        
        current_time = rospy.get_time()
        elapsed_time = current_time - prev_time
        prev_time = current_time

        control_frequency = 1.0 / 0.01
        q = q + q_dot / control_frequency

        t2_check = rospy.get_time()
        q = check_joints_position(q, use_wrist)
        q_dot = check_joints_velocity(q_dot, use_wrist)

        t1_check = rospy.get_time()
        time_check = t1_check - t2_check

        x += 1 / control_frequency + time_check

        arm_str = JointTrajectory()
        arm_str.header = Header()
        arm_str.header.stamp = rospy.Time.now()
        arm_str.joint_names = JOINT_NAMES_ARM
        arm_point = JointTrajectoryPoint(
            positions=q[:7, 0].tolist(),
            velocities=q_dot[:7, 0].tolist(),
            accelerations=[0] * 7,
            time_from_start=rospy.Duration(x)
        )
        arm_str.points.append(arm_point)
        pub_arm.publish(arm_str)

        if use_wrist:
            wrist_str = JointTrajectory()
            wrist_str.header = Header(stamp=rospy.Time.now())
            wrist_str.joint_names = ['qbmove2_shaft_joint', 'qbmove2_stiffness_preset_virtual_joint']
            wrist_point = JointTrajectoryPoint(
                positions=[q[7, 0], STIFFNESS_MAX],
                velocities=[q_dot[7, 0], 0.0],
                accelerations=[0, 0],
                time_from_start=rospy.Duration(x)
            )
            wrist_str.points.append(wrist_point)
            pub_wrist.publish(wrist_str)

        q = check_q_goal(q, use_wrist)

        norm_e_t = np.linalg.norm(e[:3, :], 2)
        norm_e_o = np.linalg.norm(e[3:, :], 2)

        rospy.loginfo(f"Translation error norm: {norm_e_t}")
        rospy.loginfo(f"Orientation error norm: {norm_e_o}")

        if norm_e_t < ERROR_TRANSLATION_THRESHOLD and norm_e_o < ERROR_ORIENTATION_THRESHOLD:
            rospy.loginfo("Visual servoing completed!")
            return True

if __name__ == '__main__':
    rospy.init_node('controller')
    listener = tf.TransformListener()

    use_wrist = rospy.get_param('~use_wrist', False)

    rospy.loginfo("Starting grasp phase")
    response = call_service("grasp_tool_task", SetBool, True)
    if response:
        rospy.loginfo(response.message)

    rospy.loginfo("Starting visual servoing")
    if servoing(use_wrist):
        rospy.loginfo("Starting pick and throw phase")
        response = call_service("place_tool_task", SetBool, True)
        if response:
            rospy.loginfo(response.message)
        rospy.loginfo("Task completed!")
    else:
        rospy.logerr("Visual servoing failed!")