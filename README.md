# MIND-Stack: Modular, Interpretable, End-to-End Differentiability for Autonomous Robots

This README file accompanies the paper "MIND-Stack: Modular, Interpretable, End-to-End Differentiability for Autonomous Robots". It provides instructions on how to install and run the code, along with explanations of the different Python files and other resources included in the repository.

## Installation

The code runs on Ubuntu 20 and ROS2 Foxy due to the NVIDIA Jetson's software. The stack is tested on a desktop with an AMD CPU and NVIDIA 4090 GPU, as well as a laptop with an Intel CPU and NVIDIA 3070 GPU.

To run the code, you need to install the following:

1. **F1Tenth Gym**: Install it from [here](https://github.com/f1tenth/f1tenth_gym).
2. **CUDA**: A working installation of CUDA is required. The appropriate CUDA version depends on your GPU. For easier installation, we recommend using the [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software).
3. **Other Dependencies**: Install the remaining dependencies using the provided `requirements.txt` file:
pip3 install -r requirements.txt

**Note**: Installing packages can sometimes cause compatibility issues. We recommend installing inside a conda or virtual environment. If you encounter problems during installation, please check the error message to identify which package requires manual installation or which versions cause mismatches.

## Usage

Some results are written to TensorBoard. To open TensorBoard, run the following command in the folder where the logs are saved:
tensorboard --log-dir=./

## File Descriptions

### Map and Waypoints

- **Map.pgm**: Represents the occupancy grid map, where white pixels represent drivable areas, black pixels obstacles, and grey pixels unknown states.
- **Map.yaml**: Contains parameters such as the resolution, origin, and start position on the map necessary to convert between pixels and meters and vice versa.
- **Map.csv**: Contains all necessary values for the waypoints, including the pose (x, y, θ) and the velocity of each point.

### Python Files

- **stanley.py**: Implements the Stanley controller with fully-differentiable calculations of the cross-track and heading error. The parameters `k_e` (line 29) and `k_h` (line 30) can be changed to try out different combinations.
- **localization.py**: Implements the CNN network for the localization module. To enable compatibility with batched input for training, make the following changes:
  - Uncomment line 79.
  - Comment line 80.
  - Comment line 94.
  - Comment the `[0]` in line 109.
  Revert these changes to run inference again.

### Accompanying Files

- **dataset_generator.py**: Creates the training dataset with LiDAR-scan pose pairs for the localization network. Adjust the number of dataset augmentations as needed.
- **train_localization.py**: Implements the training algorithm for the localization network. You can choose the number of epochs for training. The pretrained model `Base_Localization_Model.pth` is provided.

### Motion Model

- **motion_model.py**: Implements the double kinematic pose propagation to connect control output to the error metric, enabling training of the localization module on the control loss.

### Training and Inference

- **train.py**: Implements end-to-end training of the stack. Users can:
  - Include or remove loss terms (lines 156-160).
  - Change the number of laps for training (line 153).
  - Set the run name for TensorBoard (line 168).
  - Choose the base localization model (line 169).
  - Visualize the ground truth pose for laps 1, 16, and 31 by commenting/uncommenting lines 272, 277, and 282 respectively.
  
  By default we conduct the first experiment from the paper reducing the cross-track error.

- **inference.py**: Runs and visualizes the network without training. Choose the network model in line 108. Visualize the blue pose estimate (line 142) and red ground truth pose (line 147) for the first lap by commenting/uncommenting the respective lines. By default the function only visualizes the ground truth pose.

### Gradient Flow Graph

- The folder `Gradient Flow Graph` contains a PNG visualizing the entire end-to-end gradient flow between the control loss and localization module. This image is also created while running `train.py`.

### ROS2 Code. 
The `ROS2` Folder contains the entire code base, reimplementing the algorithm for the ROS2 framework for use on the F1Tenth vehicle and NVIDIA Jetson.
We utilize the standard Particle Filter found [here](https://github.com/f1tenth/particle_filter).
