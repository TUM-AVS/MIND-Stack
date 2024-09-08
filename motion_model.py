import torch
import torch.nn as nn

class Motion_Model(nn.Module):
    """
    Implements a fully differentiable motion model. It propagates the pose estimate for two timesteps (each 0.01s). 
    The resulting position is used to calculate the cross-track and heading error for the control loss. 
    This enables end-to-end gradient flow, allowing the localization module to be trained on the control loss.
    """
    def __init__(self):
        super(Motion_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize vehicle parameters
        self.lf = 0.15875
        self.lr = 0.17145
        self.lwb = self.lf + self.lr
        self.max_steering_angle = 0.4189

        # Set timesteps for each propagation
        self.time_step1 = 0.01
        self.time_step2 = 0.01

    def forward(self, pose, velocity, steer):
        # Propagate the pose (x, y, orientation (theta)) for the first timestep
        pose1_x = pose[0] + torch.cos(pose[2]) * velocity * self.time_step1
        pose1_y = pose[1] + torch.sin(pose[2]) * velocity * self.time_step1
        pose1_theta = torch.remainder(pose[2] + torch.tan(steer) * self.time_step1 * velocity / self.lwb + torch.pi, 2 * torch.pi) - torch.pi

        # Propagate the pose for the second timestep
        pose2_x = pose1_x + torch.cos(pose1_theta) * velocity * self.time_step2
        pose2_y = pose1_y + torch.sin(pose1_theta) * velocity * self.time_step2
        pose2_theta = torch.remainder(pose1_theta + torch.tan(steer) * self.time_step2 * velocity / self.lwb + torch.pi, 2 * torch.pi) - torch.pi

        # Return the propagated pose for subsequent calculation of the control loss
        return torch.stack([pose2_x, pose2_y, pose2_theta]), pose1_theta