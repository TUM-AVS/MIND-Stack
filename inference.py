import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from pyglet.gl import GL_POINTS
from OpenGL.GL import glPointSize
import torch

# Load the Stanley Controller and the Localization Model from their respective Python files
from stanley import DiffStanleyController
from localization import LocalizationModel1D, PositionalEncoding

"""
This code performs pure inference without any training. It allows the user to choose the localization model
to test and visualizes both the ground truth pose and the localization model output. 
The code uses a Differential Stanley controller for path following in the F1tenth gym environment.
"""

def render_waypoints(conf, waypoints, drawn_waypoints, e):
    """
    Update waypoints being drawn by EnvRenderer.
    """
    points = np.vstack((waypoints[:, conf.wpt_xind].cpu(), waypoints[:, conf.wpt_yind].cpu())).T
    scaled_points = 50. * points
    glPointSize(2.5)

    for i in range(points.shape[0]):
        if len(drawn_waypoints) < points.shape[0]:
            b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                            ('c3B/stream', [0, 0, 0]))
            drawn_waypoints.append(b)
        else:
            drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

def inference():
    """
    Main entry point for inference.
    """
    # Set the device for CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration and waypoints
    with open('Map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    DiffStanley = DiffStanleyController()
    waypoints_np = np.loadtxt('Map.csv', delimiter=';', skiprows=3)
    waypoints = torch.tensor(waypoints_np, dtype=torch.float32, device=device, requires_grad=False)
    waypoints[:, 3] = waypoints[:, 3] - torch.pi / 2
    waypoints[:, 3] = torch.remainder(waypoints[:, 3], 2 * torch.pi)
    
    # Initialize the arrays for rendering
    positions_to_render = []
    positions_to_render_gt = []
    positions_to_render2 = []
    drawn_waypoints = []

    def render_callback(env_renderer):
        """
        Custom drawing function.
        Args:
            env_renderer: Environment renderer instance.
        """
        e = env_renderer

        # Update camera to follow the car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700

        render_waypoints(conf, waypoints, drawn_waypoints, env_renderer)

        if obs['lap_counts'] == 0:
            for pos in positions_to_render:
                e.batch.add(1, GL_POINTS, None, 
                            ('v3f/stream', [pos[0], pos[1], 0.0]), 
                            ('c3B/stream', [0, 0, 255]))
                
        if obs['lap_counts'] == 0:
            for pos in positions_to_render_gt:
                e.batch.add(1, GL_POINTS, None, 
                            ('v3f/stream', [pos[0], pos[1], 0.0]), 
                            ('c3B/stream', [255, 0, 0]))
            
        for pos in positions_to_render2:
            e.batch.add(1, GL_POINTS, None, 
                        ('v3f/stream', [pos[0], pos[1], 0.0]), 
                        ('c3B/stream', [0, 255, 0]))

    # Initialize the environment
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Initialize the previous pose estimate for the localization module
    previous_pose = torch.tensor(np.array([[conf.sx, conf.sy, conf.stheta]]), dtype=torch.float32, device=device)

    laptime = 0.0
    start = time.time()

    # Load the pre-trained localization model
    ConditionalCNN = LocalizationModel1D()
    ConditionalCNN = torch.load('Base_Localization_Model.pth')
    ConditionalCNN.to(device)

    # Parameters for debugging and additional features
    lap_counter = 0
    max_speed = 0.0
    avg_laptime = 0.0

    while not done:
        # Perform inference and update pose estimate
        lidar_scan = torch.tensor(obs['scans'][0], dtype=torch.float32, device=device)
        pose_estimate = ConditionalCNN.forward(lidar_scan, previous_pose)

        steer, velocity, crosstrack_error, heading_error = DiffStanley.forward(pose_estimate)
        steer_stanley_np = steer.cpu().detach().numpy()
        speed_stanley_np = velocity.cpu().detach().numpy()

        # Perform the action in the environment
        obs, step_reward, done, info = env.step(np.array([[steer_stanley_np, 1.0 * speed_stanley_np]]))
        env.render()

        # Keep the loop running, the F1Tenth Gym offers multiple options to end the simulation
        done = False

        # Save the maximum velocity achieved by the vehicle
        if obs['linear_vels_x'][0] > max_speed:
            max_speed = obs['linear_vels_x'][0]

        laptime += step_reward
        previous_pose = pose_estimate

        # Draw the pose estimate from the localization module in blue
        if obs['lap_counts'] == 0:
            drawn_motion_model_benchmark_pose = 50. * pose_estimate[:2].detach().cpu().numpy()
            # positions_to_render.append((drawn_motion_model_benchmark_pose[0], drawn_motion_model_benchmark_pose[1]))

        # Draw the ground truth pose from the localization module in red
        if obs['lap_counts'] == 0:
            drawn_motion_model_benchmark_pose = 50. * np.array([obs['poses_x'][0], obs['poses_y'][0]])
            positions_to_render_gt.append((drawn_motion_model_benchmark_pose[0], drawn_motion_model_benchmark_pose[1]))

        # After each completed lap, print the laptime and average laptime
        if obs['lap_counts'] == lap_counter + 1:
            lap_counter += 1
            print('Laptime:', laptime)
            avg_laptime += laptime
            print('Average Laptime:', avg_laptime / obs['lap_counts'])
            laptime = 0.0

        # End the simulation after 30 laps
        if obs['lap_counts'] == 30:
            break

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    print('Max speed:', max_speed)

if __name__ == '__main__':
    inference()