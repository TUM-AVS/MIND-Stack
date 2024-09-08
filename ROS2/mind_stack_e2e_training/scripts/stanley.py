import numpy as np
from argparse import Namespace
import torch
import torch.nn as nn

class DiffStanleyController(nn.Module):
    """
    Implements the hand-crafted, rule-based Stanley Controller.
    """
    def __init__(self):
        super(DiffStanleyController, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set constant velocity parameter
        self.velocity_goal = 0.9

        # Initialize vehicle parameters
        self.lf = 0.15875
        self.lr = 0.17145
        self.max_steering_angle = 0.4189

        # Load the waypoints
        waypoints_np = np.loadtxt('Map.csv', delimiter=';', skiprows=3)
        self.waypoints = torch.tensor(waypoints_np, dtype=torch.float32, device=self.device, requires_grad=False)
        self.waypoints[:, 3] = self.waypoints[:, 3] - torch.pi / 2
        self.waypoints[:, 3] = torch.remainder(self.waypoints[:, 3], 2 * torch.pi)

        # Set the Stanley parameters
        self.k_e = nn.Parameter(torch.tensor(1.8, dtype=torch.float32, device=self.device), requires_grad=False)
        self.k_h = nn.Parameter(torch.tensor(1.3, dtype=torch.float32, device=self.device), requires_grad=False)

        # Initialize the steering angle
        self.steer = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False)

    def error_model(self, pose):
        """
        Calculates the cross-track and heading error of an input pose to the closest waypoint.
        """
        # Convert the pose from the center of gravity to the front axle
        pose_cg_x = pose[0] + self.lf * torch.sin(pose[2])
        pose_cg_y = pose[1] + self.lf * torch.cos(pose[2])
        pose_cg_theta = torch.remainder(pose[2] + torch.pi, 2 * torch.pi)

        # Calculate the Euclidean distance between the pose and all waypoints
        distance_to_waypoints = torch.norm(self.waypoints[:, 1:3] - torch.stack([pose_cg_x, pose_cg_y], dim=-1), dim=1)
        min_index = torch.argmin(distance_to_waypoints)
        distance_to_waypoint = torch.stack([pose_cg_x - self.waypoints[min_index, 1], pose_cg_y - self.waypoints[min_index, 2]])
        front_axle_vector = torch.stack([-torch.cos(pose_cg_theta + torch.pi / 2), -torch.sin(pose_cg_theta + torch.pi / 2)])

        # Compute the cross-track error
        crosstrack_error = torch.dot(distance_to_waypoint, front_axle_vector)

        # Compute the heading error
        raw_heading_error = (self.waypoints[min_index, 3] - pose_cg_theta)

        # Clamp the heading error between 0 and 2pi
        heading_error = torch.remainder(raw_heading_error + torch.pi, 2 * torch.pi) - torch.pi

        return crosstrack_error, heading_error, min_index

    def forward(self, pose):
        """
        Implements the Stanley Controller for the F1Tenth car.
        """
        crosstrack_err, heading_err, min_index = self.error_model(pose)

        # Set the velocity according to the waypoints and multiply it with a factor for stability
        velocity = self.waypoints[min_index, 5] * self.velocity_goal

        # Calculate the steering angle according to the Stanley Controller's control law
        self.steer = self.k_h * heading_err + torch.atan(self.k_e * -crosstrack_err / (velocity + 0.00001))

        return self.steer, velocity, crosstrack_err, heading_err