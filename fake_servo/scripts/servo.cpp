#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <std_msgs/Header.h>
#include <std_srvs/SetBool.h>
#include <Eigen/Dense>
#include "kinematics.h"
#include "tools.h"

// Constants
const std::vector<std::string> JOINT_NAMES_ARM = {
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
    "panda_joint5", "panda_joint6", "panda_joint7"
};
const std::vector<std::string> JOINT_NAMES_WRIST = {
    "qbmove2_motor_1_joint", "qbmove2_motor_2_joint", "qbmove2_shaft_joint",
    "qbmove2_deflection_virtual_joint", "qbmove2_stiffness_preset_virtual_joint"
};
const double STIFFNESS_MAX = 1.0;
const double ERROR_TRANSLATION_THRESHOLD = 0.005;
const double ERROR_ORIENTATION_THRESHOLD = 0.01;
const double CONTROL_FREQUENCY = 100.0;

bool callService(const std::string& service_name, bool request) {
    ros::ServiceClient client = ros::NodeHandle().serviceClient<std_srvs::SetBool>(service_name);
    std_srvs::SetBool srv;
    srv.request.data = request;
    if (client.call(srv)) {
        ROS_INFO_STREAM("Service response: " << srv.response.message);
        return srv.response.success;
    } else {
        ROS_ERROR_STREAM("Failed to call service: " << service_name);
        return false;
    }
}

Eigen::VectorXd checkQGoal(const Eigen::VectorXd& q_goal, bool use_wrist) {
    ros::NodeHandle nh;
    ros::Rate rate(100);
    while (ros::ok()) {
        sensor_msgs::JointStateConstPtr joint_states = ros::topic::waitForMessage<sensor_msgs::JointState>("/joint_states", nh);
        sensor_msgs::JointStateConstPtr wrist_states;
        if (use_wrist) {
            wrist_states = ros::topic::waitForMessage<sensor_msgs::JointState>("/robot_arm/gripper/qbmove2/joint_states", nh);
        }

        if (joint_states->name == JOINT_NAMES_ARM && (!use_wrist || wrist_states->name == JOINT_NAMES_WRIST)) {
            Eigen::VectorXd q_arm_actual = Eigen::Map<const Eigen::VectorXd>(joint_states->position.data(), 7);
            Eigen::VectorXd q_actual;
            if (use_wrist) {
                Eigen::VectorXd q_wrist_actual(1);
                q_wrist_actual(0) = wrist_states->position[2];
                q_actual = Eigen::VectorXd(q_arm_actual.size() + q_wrist_actual.size());
                q_actual << q_arm_actual, q_wrist_actual;
            } else {
                q_actual = q_arm_actual;
            }

            Eigen::VectorXd e_q = q_goal - q_actual;
            if (e_q.norm() < 0.01) {
                return q_actual;
            }
        }
        rate.sleep();
    }
    return Eigen::VectorXd();
}

Eigen::VectorXd initializeState(bool use_wrist) {
    ros::NodeHandle nh;
    ros::Rate rate(100);
    while (ros::ok()) {
        sensor_msgs::JointStateConstPtr joint_states = ros::topic::waitForMessage<sensor_msgs::JointState>("/joint_states", nh);
        sensor_msgs::JointStateConstPtr wrist_states;
        if (use_wrist) {
            wrist_states = ros::topic::waitForMessage<sensor_msgs::JointState>("/robot_arm/gripper/qbmove2/joint_states", nh);
        }

        if (joint_states->name == JOINT_NAMES_ARM && (!use_wrist || wrist_states->name == JOINT_NAMES_WRIST)) {
            Eigen::VectorXd q_arm = Eigen::Map<const Eigen::VectorXd>(joint_states->position.data(), 7);
            if (use_wrist) {
                Eigen::VectorXd q_wrist(1);
                q_wrist(0) = wrist_states->position[2];
                Eigen::VectorXd q(q_arm.size() + q_wrist.size());
                q << q_arm, q_wrist;
                return q;
            } else {
                return q_arm;
            }
        }
        rate.sleep();
    }
    return Eigen::VectorXd();
}

bool servoing(bool use_wrist) {
    ros::NodeHandle nh;
    ros::Publisher pub_arm = nh.advertise<trajectory_msgs::JointTrajectory>(
        "/position_joint_trajectory_controller/command", 10
    );
    ros::Publisher pub_wrist;
    if (use_wrist) {
        pub_wrist = nh.advertise<trajectory_msgs::JointTrajectory>(
            "/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command", 10
        );
    }

    tf::TransformListener listener;
    ros::Rate rate(CONTROL_FREQUENCY);
    Eigen::VectorXd q = initializeState(use_wrist);
    double x = 0.0;

    tf::StampedTransform transform;
    while (ros::ok()) {
        try {
            listener.lookupTransform("/panda_link0", "/object", ros::Time(0), transform);
            break;
        } catch (tf::TransformException& ex) {
            ROS_WARN("%s", ex.what());
            ros::Duration(0.01).sleep();
        }
    }

    while (ros::ok()) {
        try {
            listener.lookupTransform("/panda_link0", "/tool_extremity", ros::Time(0), transform);
        } catch (tf::TransformException& ex) {
            ROS_WARN("%s", ex.what());
            ros::Duration(0.01).sleep();
            continue;
        }

        // Feedback control logic here (similar to Python implementation)
        // ...

        rate.sleep();
    }
    return false;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "controller");
    ros::NodeHandle nh;

    bool use_wrist;
    nh.param("~use_wrist", use_wrist, false);

    ROS_INFO("Starting grasp phase");
    if (callService("grasp_tool_task", true)) {
        ROS_INFO("Starting visual servoing");
        if (servoing(use_wrist)) {
            ROS_INFO("Starting pick and throw phase");
            if (callService("place_tool_task", true)) {
                ROS_INFO("Task completed!");
            }
        } else {
            ROS_ERROR("Visual servoing failed!");
        }
    }

    return 0;
}
