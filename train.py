import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from numba import njit
from pyglet.gl import GL_POINTS
from OpenGL.GL import glPointSize
import torch
import torch.optim as optim
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

# Load the Stanley Controller, Motion Model, and Localization Module
from stanley import DiffStanleyController
from localization import LocalizationModel1D, PositionalEncoding
from motion_model import Motion_Model

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

def train():
    """
    This function trains the localization module on the downstream control loss.
    """
    # Determine the device to be used (GPU if available, else CPU)
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

    # Initialize rendering arrays
    drawn_waypoints = []

    def render_callback(env_renderer):
        """
        Custom callback function for rendering.
        """
        e = env_renderer

        # Update camera to follow the car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700

        render_waypoints(conf, waypoints, drawn_waypoints, env_renderer)

        # Visualize the trajectory in different laps
        if lap == 0:
            for pos in positions_first_lap:
                env_renderer.batch.add(1, GL_POINTS, None, 
                                    ('v3f/stream', [pos[0], pos[1], 0.0]), 
                                    ('c3B/stream', [0, 101, 189]))
            
        if lap == 15:
            for pos in positions_middle_lap:
                env_renderer.batch.add(1, GL_POINTS, None, 
                                    ('v3f/stream', [pos[0], pos[1], 0.0]), 
                                    ('c3B/stream', [162, 173, 0]))

        if lap == 30:
            for pos in positions_last_lap:
                env_renderer.batch.add(1, GL_POINTS, None, 
                                    ('v3f/stream', [pos[0], pos[1], 0.0]), 
                                    ('c3B/stream', [227, 114, 34]))
            
        glPointSize(3.0)

    # Set the lap counter
    lap = 0

    # Initialize the gym environment
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)

    # Initialize the previous pose
    prev_pose = torch.tensor(np.array([[conf.sx, conf.sy, conf.stheta]]), dtype=torch.float32, device=device)

    # Initialize the pose estimate
    pose_estimate = torch.tensor(np.array([[conf.sx, conf.sy, conf.stheta]]), dtype=torch.float64, device=device)

    # Set the lap time
    laptime = 0.0
    start = time.time()
    
    # Function to convert meters to pixels
    def meters_to_pixels(x, y):
        pixel_x = int((x - conf.origin[0]) / conf.resolution)
        pixel_y = int((y - conf.origin[1]) / conf.resolution)
        return pixel_x, pixel_y

    # Initialize the localization and stanley modules
    DiffLocalization = LocalizationModel1D()
    DiffStanley = DiffStanleyController()
    loss_fn = torch.nn.MSELoss()

    # Set the initial minimum loss and maximum speed
    minimum_loss = 1000000.0
    max_speed = 0.0

    # Initialize previous heading angles for orientation loss calculation
    previous_heading_angle = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
    second_previous_heading_angle = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)

    lap_visualizer = 0

    # Set initial values for different loss metrics
    min_localization_loss = 1000000.0
    max_localization_loss = 0.0
    avg_localization_loss_per_lap = 0.0

    min_cross_error = 1000000.0
    max_cross_error = 0.0
    avg_cross_error_per_lap = 0.0

    min_head_error = 1000000.0
    max_head_error = 0.0
    avg_head_error_per_lap = 0.0

    min_orient_error = 1000000.0
    max_orient_error = 0.0
    avg_orient_error_per_lap = 0.0

    avg_gt_head_error_per_lap = 0.0
    avg_gt_cross_error_per_lap = 0.0

    num_laps_avg_loss = 1

    # Set the number of laps the simulation runs
    num_laps = 31

    # Set hyperparameters for the loss function
    localization_weight = 0.0
    control_cross_weight = 5.5
    control_heading_weight = 0.0
    control_orient_weight = 1.0
    learning_rate = 0.00000009

    # Initialize arrays for pose visualizations
    positions_first_lap = []
    positions_middle_lap = []
    positions_last_lap = []

    # Start a Tensorboard writer, set a run name, and the localization module to be loaded
    run_name = '1'
    localization_model_name = 'Base_Localization_Model.pth'
    writer = SummaryWriter(run_name)

    # Add the hyperparameters to Tensorboard
    writer.add_hparams({'localization_weight': localization_weight, 'control_cross_weight': control_cross_weight, 'control_heading_weight': control_heading_weight, 'control_orient_weight': control_orient_weight, 'learning_rate': learning_rate, 'Stanley_K_e': DiffStanley.k_e.item(), 'Stanley_K_h': DiffStanley.k_h.item(), 'Stanley_velocity_perc': DiffStanley.velocity_goal, 'Localization_Model': localization_model_name}, {})

    # Load the previously best localization model
    DiffLocalization = torch.load(localization_model_name)
    optimizer_localization = optim.Adam(DiffLocalization.parameters(), lr=learning_rate)

    # Load the motion model
    motion_model = Motion_Model()

    # Iterate over the set number of laps
    for lap in range(num_laps):
        print('Lap:', lap)
        
        # Reset the vehicle to its starting position
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        
        # Reset variables for each lap
        lap_time = 0.0
        lap_reward = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_localization_reward = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_cross_error = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_head_error = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_orient_error = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_cross_error_gt = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_head_error_gt = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        lap_local_error_gt = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        
        # Set the optimizer grad to zero
        optimizer_localization.zero_grad()

        # Reset crashed parameter if the vehicle crashed the lap before
        crashed = False
        checked = 1
        count_finished_lap = 1

        # Run training for more than one lap before resetting the vehicle if needed
        while obs['lap_counts'] < 1:
            while obs['lap_counts'] < checked:

                # Debug set for PyTorch
                torch.autograd.set_detect_anomaly(True)

                # Forward pass through the localization model
                lidar_scan = torch.tensor(obs['scans'][0], dtype=torch.float32, device=device)
                pose_estimate = DiffLocalization.forward(lidar_scan, prev_pose)

                # Save the pose estimate
                prev_pose = pose_estimate.clone().detach()

                # Forward pass through the control model
                steer, velocity, crosstrack_error_orig, heading_error_orig = DiffStanley.forward(pose_estimate)

                # Convert control values from tensor to numpy
                steer_stanley_np = steer.cpu().detach().numpy()
                speed_stanley_np = velocity.cpu().detach().numpy()

                # Step the environment for one timestep
                obs, step_reward, done, info = env.step(np.array([[steer_stanley_np, speed_stanley_np]]))

                # Save the ground truth pose
                real_pose = torch.tensor([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]], dtype=torch.float32, device=device, requires_grad=False)

                # Use the motion model to predict the next two poses, based on the real pose
                motion_model_pose, pose1_theta = motion_model.forward(real_pose, steer, velocity)

                # Calculate the control loss based on the propagated pose
                crosstrack_error, heading_error, min_index = DiffStanley.error_model(motion_model_pose)

                # Calculate the localization loss and sum it up per lap
                localization_loss = localization_weight * loss_fn(pose_estimate, torch.tensor(np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]), dtype=torch.float32, device=device))
                lap_localization_reward = lap_localization_reward + localization_loss
                
                # Calculate the control losses and sum them up per lap
                control_loss_orient = control_orient_weight * torch.abs(pose1_theta - 2 * previous_heading_angle + second_previous_heading_angle)
                control_loss_cross = control_cross_weight * (crosstrack_error)**2
                controll_loss_head = - control_heading_weight * (heading_error)
                lap_cross_error = lap_cross_error + control_loss_cross
                lap_head_error = lap_head_error + controll_loss_head
                lap_orient_error = lap_orient_error + control_loss_orient

                # Calculate the total loss as the sum of all individual losses
                total_loss = control_loss_orient + control_loss_cross + localization_loss + controll_loss_head

                # Calculate the ground truth error
                crosstrack_error_gt, heading_error_gt, min_index = DiffStanley.error_model(real_pose)
                lap_cross_error_gt = lap_cross_error_gt + crosstrack_error_gt
                lap_head_error_gt = lap_head_error_gt + heading_error_gt
                lap_local_error_gt = lap_local_error_gt + loss_fn(pose_estimate, torch.tensor(np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]), dtype=torch.float32, device=device))

                # Train the localization module based on the total loss
                optimizer_localization.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(DiffLocalization.parameters(), max_norm=1.0)
                optimizer_localization.step()

                # Visualize the difference in the poses in the 1st, 16th and 31st lap
                if lap == 0:
                    real_pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
                    drawn_motion_model_benchmark_pose2 = 50.*real_pose
                    positions_first_lap.append((drawn_motion_model_benchmark_pose2[0], drawn_motion_model_benchmark_pose2[1]))

                if lap == 15:
                    real_pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
                    drawn_motion_model_benchmark_pose2 = 50.*real_pose
                    positions_middle_lap.append((drawn_motion_model_benchmark_pose2[0], drawn_motion_model_benchmark_pose2[1]))

                if lap == 30:
                    real_pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
                    drawn_motion_model_benchmark_pose2 = 50.*real_pose
                    positions_last_lap.append((drawn_motion_model_benchmark_pose2[0], drawn_motion_model_benchmark_pose2[1]))

                # Save the two previous orientations
                previous_heading_angle = pose1_theta.clone().detach()
                second_previous_heading_angle = previous_heading_angle.clone().detach()

                # Save the maximum speed
                if obs['linear_vels_x'][0] > max_speed:
                    max_speed = obs['linear_vels_x'][0]

                # Create a gradient flow image in the first timestep
                if lap_visualizer == 0:
                    named_params = {**dict(DiffLocalization.named_parameters()), **dict(DiffStanley.named_parameters())}
                    dot = make_dot(total_loss, params=named_params)
                    dot.format = 'png'
                    dot.render('computation_graph_with_names')
                    lap_visualizer = 1

                # Summarize the total loss over one lap
                lap_reward = lap_reward + total_loss

                # Save the lap time
                lap_time += step_reward
                laptime += step_reward

                # Render the state of the simulation environment
                env.render()

                # If we finish one lap, check if the current lap was the best overall and save this model
                if obs['lap_counts'] == count_finished_lap:
                    if not crashed:
                        # Save the overall best model
                        if torch.abs(lap_reward) < minimum_loss:
                            checkpoint = {
                            'model_localization_state_dict': DiffLocalization.state_dict(),
                            'optimizer_localization_state_dict': optimizer_localization.state_dict(),
                            }
                            torch.save(checkpoint, 'checkpoint_best.pth')
                            torch.save(DiffLocalization, 'DiffLocalization_best_' + str(run_name) + '.pth')
                            print("--------------------")
                            print("Saving the best model")
                            print("--------------------")
                            minimum_loss = lap_time

                        # Print different evaluation metrics for the past lap
                        print("Laptime: ", lap_time)
                        writer.add_scalar("Loss/Lap_Time", lap_time, lap)
                        print("Lap reward: ", lap_reward)
                        writer.add_scalar("Loss/Lap_Reward", lap_reward, lap)
                        print("Localization loss: ", lap_localization_reward.item())
                        writer.add_scalar("Loss/Localization_Loss", lap_localization_reward, lap)
                        print("Cross-track error: ", lap_cross_error.item())
                        writer.add_scalar("Loss/Crosstrack_Error", lap_cross_error, lap)
                        print("Heading error: ", lap_head_error.item())
                        writer.add_scalar("Loss/Heading_Error", lap_head_error, lap)
                        print("Orientation error: ", lap_orient_error.item())
                        writer.add_scalar("Loss/Orientation_Error", lap_orient_error, lap)
                        print("Ground truth cross-track error: ", lap_cross_error_gt.item())
                        writer.add_scalar("Loss/Crosstrack_Error_GT", lap_cross_error_gt, lap)
                        print("Ground truth heading error: ", lap_head_error_gt.item())
                        writer.add_scalar("Loss/Heading_Error_GT", lap_head_error_gt, lap)
                        print("Ground truth localization loss: ", lap_local_error_gt.item())
                        writer.add_scalar("Loss/Localization_Loss_GT", lap_local_error_gt, lap)

                        # Calculate the average loss values over all laps
                        avg_localization_loss_per_lap += lap_localization_reward.item()
                        avg_cross_error_per_lap += lap_cross_error.item()
                        avg_head_error_per_lap += lap_head_error.item()
                        avg_orient_error_per_lap += lap_orient_error.item()
                        avg_gt_cross_error_per_lap += lap_cross_error_gt.item()
                        avg_gt_head_error_per_lap += lap_head_error_gt.item()

                        # Print the average loss values over all laps
                        print("Average localization loss per lap: ", avg_localization_loss_per_lap / num_laps_avg_loss)
                        print("Average cross-track error per lap: ", avg_cross_error_per_lap / num_laps_avg_loss)
                        print("Average heading error per lap: ", avg_head_error_per_lap / num_laps_avg_loss)
                        print("Average orientation error per lap: ", avg_orient_error_per_lap / num_laps_avg_loss)
                        print("Average ground truth cross-track error per lap: ", avg_gt_cross_error_per_lap / num_laps_avg_loss)
                        print("Average ground truth heading error per lap: ", avg_gt_head_error_per_lap / num_laps_avg_loss)
                        num_laps_avg_loss += 1

                        # Save the minimum and maximum loss values
                        if lap_localization_reward.item() < min_localization_loss:
                            min_localization_loss = lap_localization_reward.item()
                        if lap_localization_reward.item() > max_localization_loss:
                            max_localization_loss = lap_localization_reward.item()
                        if lap_cross_error.item() < min_cross_error:
                            min_cross_error = lap_cross_error.item()
                        if lap_cross_error.item() > max_cross_error:
                            max_cross_error = lap_cross_error.item()
                        if lap_head_error.item() < min_head_error:
                            min_head_error = lap_head_error.item()
                        if lap_head_error.item() > max_head_error:
                            max_head_error = lap_head_error.item()
                        if lap_orient_error.item() < min_orient_error:
                            min_orient_error = lap_orient_error.item()
                        if lap_orient_error.item() > max_orient_error:
                            max_orient_error = lap_orient_error.item()

                        # Calculate the difference between the maximum and minimum losses
                        print("Max difference in localization loss: ", max_localization_loss - min_localization_loss)
                        print("Max difference in cross-track error: ", max_cross_error - min_cross_error)
                        print("Max difference in heading error: ", max_head_error - min_head_error)
                        print("Max difference in orientation error: ", max_orient_error - min_orient_error)

                        # Reset variables for the next lap
                        lap_time = 0.0
                        lap_reward = 0.0
                        lap_localization_reward = 0.0
                        lap_cross_error = 0.0
                        lap_head_error = 0.0
                        lap_orient_error = 0.0
                        lap_reward = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False) 
                        lap_localization_reward = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                        lap_cross_error = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                        lap_head_error = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                        lap_orient_error = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                        lap_cross_error_gt = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                        lap_head_error_gt = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                        lap_local_error_gt = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
                    count_finished_lap += 1
                    writer.flush()

                if obs['lap_counts'] == 10000:
                    break

            if crashed:
                break

            # Save a model checkpoint every 5 laps, overwriting the previous one
            if crashed == False and checked == obs['lap_counts'] and obs['lap_counts'] % 5 == 0:
                checked += 5
                checkpoint = {
                    'model_localization_state_dict': DiffLocalization.state_dict(),
                    'optimizer_localization_state_dict': optimizer_localization.state_dict(),
                }
                torch.save(checkpoint, 'checkpoint.pth')
                torch.save(DiffLocalization, 'DiffLocalization_checkpoint_' + str(obs['lap_counts']) + '.pth')
                print(f'Checkpoint saved at lap {lap}')

            # Save a model checkpoint every 10 laps, without overwriting the previous one
            if lap % 10 == 0:
                Save_name = run_name + '_' + str(lap) + '.pth'
                torch.save(DiffLocalization, Save_name)

        if crashed:
            break

        writer.close()
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
        print('Max speed:', max_speed)

if __name__ == '__main__':
    train()