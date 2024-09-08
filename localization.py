import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for the previous pose. The result is combined with 
    the CNN output in the fully connected layers.
    """

    def __init__(self, L=10):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, pose):
        encoded = []
        for i in range(self.L):
            # Compute sine and cosine for each pose and append to the encoded list
            encoded.append(torch.sin((2**i) * torch.pi * pose))
            encoded.append(torch.cos((2**i) * torch.pi * pose))
        # Concatenate all the encoded values along the last dimension
        return torch.cat(encoded, dim=-1)

class LocalizationModel1D(nn.Module): 
    def __init__(self):
        super(LocalizationModel1D, self).__init__()

        """
        Initializes the localization module, implementing a 1D CNN.
        """

        # Determine the device to be used (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the positional encoder with 10 frequency components
        self.pos_encoder = PositionalEncoding(L=10)
        
        # Define a CNN with several 1D convolutional layers followed by ReLU activation and MaxPooling
        self.simple_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        ).to(self.device)

        # Fully connected layers to process the combined features
        self.fc1 = nn.Linear(512*16, 120)
        self.fc2 = nn.Linear(120 + 60, 60)
        self.fc3 = nn.Linear(60, 3)

    def forward(self, lidar_scan, previous_pose):
        """
        Performs the forward pass for inference or training of the Localization Model.
        """

        # Process the LiDAR scan

        # Unsqueeze to add batch and channel dimensions, convert to float
        # Comment the following line for inference and the second line for training
        # lidar_scan = lidar_scan.unsqueeze(1).float()
        lidar_scan = lidar_scan.unsqueeze(0).unsqueeze(0).float()

        # Pass through the CNN
        lidar_scan = self.simple_cnn(lidar_scan)
        # Flatten the output from the CNN
        lidar_scan = lidar_scan.view(lidar_scan.size(0), -1)

        # Process the previous pose and convert to grid representation
        previous_pose = previous_pose.clone().detach().float()
        grid_representation = torch.round(previous_pose / 1) * 1
        # Encode the grid representation using the positional encoder
        encoded_pose = self.pos_encoder(grid_representation)
        # Reshape encoded pose for compatibility in combining inputs
        # Comment the following line for training
        encoded_pose = encoded_pose.view(1, -1)

        # Combine features from the LiDAR scan and positional encoding
        # Pass through the first fully connected layer with ReLU activation
        lidar_scan = F.relu(self.fc1(lidar_scan))
        # Concatenate the LiDAR features and the encoded pose
        combined_input = torch.cat((lidar_scan, encoded_pose), dim=1)
        # Pass through the second fully connected layer with ReLU activation
        combined_input = F.relu(self.fc2(combined_input))

        # Final output through the last fully connected layer
        output = self.fc3(combined_input)

        # Return the output to MIND-Stack for subsequent processing by the Stanley controller
        # Comment the [0] for training
        return output[0]