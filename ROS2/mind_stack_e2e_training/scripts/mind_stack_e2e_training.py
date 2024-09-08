#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from localization import LocalizationModel1D, PositionalEncoding
from stanley import DiffStanleyController
from motion_model import Motion_Model
import time
from f110_gym.envs.laser_models import ScanSimulator2D
from tf_transformations import euler_from_quaternion
from torchviz import make_dot

class MIND_Stack_Node(Node):
    """
    MIND_Stack_Node class is a ROS2 node that implements the MIND Stack.
    """
    def __init__(self):
        super().__init__('MIND_Stack_Node')

        self.device = torch.device("cuda")
        
        # Load the pre-trained localization model
        self.DiffLocalization = torch.load('Base_Localization_Model.pth')
        self.DiffLocalization.to(self.device)

        # Initialize the Stanley controller
        self.DiffStanley = DiffStanleyController()

        # Create ROS subscribers and publishers
        self.scan_subscription = self.create_subscription( # Subscribe to the particle filter pose topic
            Odometry,
            '/pf/pose/odom',
            self.pf_odom_callback,
            10
        )

        # Initialize pose, speed, and steering angle variables
        self.init_pose = torch.tensor([0.0, 0.0, 0.0], dtype = torch.float32, device = self.device)
        self.init_speed = 0.0
        self.init_steer = 0.0

        # Store the previous pose for use in the localization model
        self.previous_pose = torch.tensor([0.0, 0.0, 0.0], dtype = torch.float32, device = self.device)
        
        # Publisher for drive commands
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 1000)

        # Publisher for the current pose as an odometry message
        self.odom_publisher_ = self.create_publisher(Odometry, '/odom_own', 1000)
        
        # Set parameters for the raytracing simulator
        self.fov = 4.7
        self.num_beams = 1080
        self.map_path = 'Map.yaml'
        self.map_ext = '.pgm'
        self.scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        self.scan_sim.set_map(self.map_path, self.map_ext)
        self.scan_rng = np.random.default_rng(seed = 12345)
        
        # Load the motion model
        self.motion_model = Motion_Model()
        
        # Set parameters for model training
        self.optimizer_local = optim.Adam(self.DiffLocalization.parameters(), lr=0.0000005)

        self.lap_visualizer = 0
        
        # Initialize previous heading angles for orientation loss calculation
        self.previous_heading_angle = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False)
        self.second_previous_heading_angle = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False)

        # Initialize error accumulators for the current lap
        self.crosstrack_gt_lap_error = 0.0
        self.heading_gt_lap_error = 0.0
        self.crosstrack_lap_error = 0.0
        self.heading_lap_error = 0.0
        
        self.init_timer = False
        self.next_time = 0.0


    def pf_odom_callback(self, pf_msg):
        """
        Callback function for the particle filter odometry.
        This function processes the incoming pose message, performs inference using the localization model,
        calculates control commands, and publishes them.
        """
        # Start timing for performance measurement
        time_start = time.time()
        drive_msg = AckermannDriveStamped()
        
        # Initialize the timer on the first run
        if self.init_timer == False:
            self.init_timer = time.time()
        
        # Convert the quaternion orientation to euler angles
        q = [pf_msg.pose.pose.orientation.x, pf_msg.pose.pose.orientation.y, pf_msg.pose.pose.orientation.z, pf_msg.pose.pose.orientation.w]
        euler = euler_from_quaternion(q)
        
        # Convert the particle filter pose to a simulated LiDAR scan
        pose_pf = np.array([pf_msg.pose.pose.position.x, pf_msg.pose.pose.position.y, euler[2]])
        scan = torch.tensor(self.scan_sim.scan(pose_pf, self.scan_rng), dtype = torch.float32, device = self.device)
        
        # Run the localization model on the scan to get the pose estimate
        pose_estimate = self.DiffLocalization.forward(scan, self.previous_pose)
        
        # Publish odom topic to visualize the pose estimate
        pose_estimate_np = pose_estimate.cpu().detach().numpy()
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.pose.pose.position.x = float(pose_estimate_np[0])
        odom_msg.pose.pose.position.y = float(pose_estimate_np[1])
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = np.sin(pose_estimate_np[2]/2)
        odom_msg.pose.pose.orientation.w = np.cos(pose_estimate_np[2]/2)
        self.odom_publisher_.publish(odom_msg)
        self.previous_pose = pose_estimate
        
        # Compute control commands using the Stanley controller
        steer, velocity = self.DiffStanley.forward(pose_estimate)
        steer_stanley_np = steer.cpu().detach().numpy()
        speed_stanley_np = velocity.cpu().detach().numpy()
        drive_msg.drive.speed = float(speed_stanley_np) * 0.20
        drive_msg.drive.steering_angle = float(steer_stanley_np)
        self.publisher_.publish(drive_msg)
        
        # Calculate the errors and train the model
        real_pose_tensor = torch.tensor(pose_pf, dtype = torch.float32, device = self.device)
        motion_model_pose, pose1_theta = self.motion_model.forward(real_pose_tensor, steer, velocity)
        crosstrack_error, heading_error, min_index = self.DiffStanley.error_model(motion_model_pose)
        self.crosstrack_lap_error += crosstrack_error
        self.heading_lap_error += heading_error

        # Calculate the individual components of the total loss
        control_loss_cross = 0.0 * (crosstrack_error)**2
        control_loss_head = - 3.4 * heading_error
        control_loss_orient = 1.0 * torch.abs(pose1_theta - 2 * self.previous_heading_angle + self.second_previous_heading_angle)

        # Combine the losses to get the total loss
        total_loss = control_loss_head + control_loss_orient + control_loss_cross

        # Perform backpropagation to update the localization model
        self.optimizer_local.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.DiffLocalization.parameters(), max_norm=1.0)
        self.optimizer_local.step()
        
        # Update previous heading angles for the next iteration
        self.previous_heading_angle = pose1_theta.clone().detach()
        self.second_previous_heading_angle = self.previous_heading_angle.clone().detach()
        
        # Calculate ground truth errors for comparison
        crosstrack_error_gt, heading_error_gt, min_index_gt = self.DiffStanley.error_model(real_pose_tensor)
        self.crosstrack_gt_lap_error += crosstrack_error_gt
        self.heading_gt_lap_error += heading_error_gt
        
        # Print errors at the end of each lap
        if min_index == 1 and time.time() > self.init_timer:
            print("-----------------------------------")
            print("The gt lap error is: ", self.crosstrack_gt_lap_error, "The gt heading error is: ", self.heading_gt_lap_error)
            self.crosstrack_gt_lap_error = 0.0
            self.heading_gt_lap_error = 0.0
            print("The lap error is: ", self.crosstrack_lap_error, "The heading error is: ", self.heading_lap_error)
            self.crosstrack_lap_error = 0.0
            self.heading_lap_error = 0.0
            self.init_timer = time.time() + 5.0

        # Visualize the gradient flow at the first lap
        if self.lap_visualizer == 0:
            named_params = {**dict(self.DiffLocalization.named_parameters()), **dict(self.DiffStanley.named_parameters())}
            dot = make_dot(control_loss_cross, params=named_params)
            dot.format = 'png'
            dot.render('computation_graph_with_names')
            self.lap_visualizer = 1

def main(args=None):
    """
    Main function to initialize and spin the ROS2 node.
    """
    rclpy.init(args=args)
    node = MIND_Stack_Node()
    rclpy.spin(node)
    MIND_Stack_Node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
