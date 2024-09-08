import yaml
import numpy as np
from argparse import Namespace
import torch
from PIL import Image
import yaml
from matplotlib import pyplot as plt
from f110_gym.envs.laser_models import ScanSimulator2D

def Create_Dataset():
    # Load the configuration file
    with open('Map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Set parameters for the raytracing simulation
    fov = 4.7  # Field of view for the LiDAR
    num_beams = 1080  # Number of LiDAR beams
    map_path = "./Map.yaml"
    map_ext = ".pgm"
    scan_sim = ScanSimulator2D(num_beams, fov)  # Initialize the ScanSimulator
    scan_sim.set_map(map_path, map_ext)  # Load the map into the simulator
    scan_rng = np.random.default_rng(seed=12345)  # Initialize the random number generator

    # Read in the map image
    map_path = "./Map.pgm"
    map_img = np.array(Image.open(map_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float32)

    # Set all non-white pixels to black
    map_img[map_img < 254] = 0

    # Visualize the map if needed
    plt.imshow(map_img, cmap='gray')
    plt.show()

    # Count the number of white pixels in the map (driveable area)
    count = 0
    for x in range(map_img.shape[1]):
        for y in range(map_img.shape[0]):
            if map_img[y, x] >= 254:
                count += 1
                plt.plot(x, y, 'ro')
    print("The number of white pixels in the map is: ", count)

    # Create tensors to store the poses and LiDAR scans for each pose on the track
    poses = torch.zeros((count, 3), dtype=torch.float32, device='cuda:0')
    scans = torch.zeros((count, num_beams), dtype=torch.float32, device='cuda:0')

    # Augment the dataset by adding more poses and scans for different orientations
    num_extensions = 10
    poses_extended = torch.zeros((count * num_extensions, 3), dtype=torch.float32, device='cuda:0')
    scans_extended = torch.zeros((count * num_extensions, num_beams), dtype=torch.float32, device='cuda:0')

    # Create LiDAR scans for each pose on the track and convert poses from pixels to meters
    for i in range(num_extensions):
        print("We are in the ", i, "th extension")
        count = 0
        for x in range(map_img.shape[1]):
            for y in range(map_img.shape[0]):
                if map_img[y, x] >= 254:
                    poses[count, 0] = x * conf.resolution + conf.origin[0] + torch.normal(mean=0.0, std=torch.arange(0.01))
                    poses[count, 1] = y * conf.resolution + conf.origin[1] + torch.normal(mean=0.0, std=torch.arange(0.01))
                    theta = scan_rng.uniform(0, 2 * np.pi)  # Random orientation
                    poses[count, 2] = theta
                    # Generate LiDAR scan for the current pose
                    scans[count, :] = torch.from_numpy(scan_sim.scan(np.array([poses[count, 0].cpu().numpy(), poses[count, 1].cpu().numpy(), poses[count, 2].cpu().numpy()]), scan_rng)).to('cuda:0')
                    count += 1
        
        # Store the extended poses and scans
        poses_extended[i * count:(i + 1) * count, :] = poses[:count, :]
        scans_extended[i * count:(i + 1) * count, :] = scans[:count, :]

    # Output the shapes of the extended poses and scans tensors
    print("The shape of the poses tensor is: ", poses_extended.shape)
    print("The shape of the scans tensor is: ", scans_extended.shape)

    # Save the extended poses and scans to disk
    torch.save(poses_extended, 'poses.pt')
    torch.save(scans_extended, 'scans.pt')

if __name__ == '__main__':
    Create_Dataset()