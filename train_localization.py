import time
import torch
import torch.optim as optim
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# Load the Localization Model from its python files
from localization import LocalizationModel1D, PositionalEncoding

def train_localization():
    # Determine the device to be used (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the localization model. Uncomment the second line to use a pretrained model
    ConditionalCNN = LocalizationModel1D()
    # ConditionalCNN = torch.load('ConditionalCNN_min.pth')
    ConditionalCNN.to(device)

    # Define the loss function, optimizer, and learning rate scheduler
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(ConditionalCNN.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Initialize the loss variables
    epoch_loss = 0.0
    epoch_10_loss = 0.0
    minimum_loss = 1000000.0

    # Load the poses and scans from the dataset
    poses = torch.load('poses.pt')
    scans = torch.load('scans.pt')
    poses = poses.to(device)
    scans = scans.to(device)

    # Create a tensor to store the previous poses
    previous_poses = poses.clone()

    # Add gaussian noise to the poses with a standard deviation of 0.01 around their true values to simulate the previous pose
    previous_poses[:, :] += torch.normal(mean=0.0, std=0.01, size=(poses.shape[0],3), device=device)

    # Define the batch size and number of batches
    batch_size = 512
    num_batches = poses.shape[0] // batch_size

    # Add the tensorboard writer and set base log directory
    base_log_dir = './tensorboard_logs'

    # Generate a unique run name using the current date and time
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_log_dir, run_name)

    # Initialize TensorBoard writer with the unique log directory
    writer = SummaryWriter(log_dir=log_dir)

    # Train the model for 100 epochs
    for epoch in range(100):
        # Iterate over the batches
        for i in range(num_batches):
            # Zero the gradients
            optimizer.zero_grad()

            # Compute the start and end indices for the current batch
            start = i * batch_size
            end = (i + 1) * batch_size

            # Forward pass
            pose_estimate = ConditionalCNN.forward(scans[start:end, :], previous_poses[start:end, :])
            
            # Debug prints to check the shapes
            print(f'Epoch: {epoch}, Batch: {i}')
            print(f'pose_estimate shape: {pose_estimate.shape}')
            print(f'poses[start:end, :].shape: {poses[start:end, :].shape}')
            
            loss = loss_fn(pose_estimate, poses[start:end, :])

            # Update the loss variables
            epoch_10_loss += loss.item()
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Update the learning rate
        scheduler.step(epoch_loss)

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Reset the epoch loss
        epoch_loss = 0.0

        # Save the model every 100 epochs
        if epoch % 100 == 0:
            torch.save(ConditionalCNN, 'ConditionalCNN_'+ str(epoch) + '_' + str(epoch_10_loss) +'_.pth')

        # Print the loss every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            print(f'Epoch {epoch} completed')
            print(f'Loss: {epoch_10_loss}')
            print("The learning rate is: ", optimizer.param_groups[0]['lr'])
            
            # Save the model if the loss is the minimum
            if epoch_10_loss < minimum_loss:
                minimum_loss = epoch_10_loss
                torch.save(ConditionalCNN, 'ConditionalCNN_min.pth')
            print("The minimum loss is: ", minimum_loss)
            epoch_10_loss = 0.0
        writer.flush()
    
    writer.close()

if __name__ == '__main__':
    train_localization()
