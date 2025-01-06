
import h5py
import torch
import atexit
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path, enabled_cameras=['cam_d405_rgb'], use_obj_pos=False, image_processors=None):
        """
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            enabled_cameras (list): List of cameras to enable (default is ['cam_d405_rgb']).
            use_obj_pos (bool): Whether to include object position data as input (default is False).
            transform_dict (dict): Optional dictionary of transforms for each camera.
        """
        self.hdf5_file_path = hdf5_file_path
        self.enabled_cameras = enabled_cameras
        self.use_obj_pos = use_obj_pos
        self.image_processors = image_processors if image_processors is not None else {}

        # Open the HDF5 file and precompute trial lengths
        self.h5_file = h5py.File(self.hdf5_file_path, 'r')
        self.trials = list(self.h5_file.keys())

        # Compute the cumulative sum of timesteps to build an index map
        self.index_map = []
        current_idx = 0
        for trial in self.trials:
            num_timesteps = self.h5_file[trial][self.enabled_cameras[0]].shape[0]
            self.index_map.append((current_idx, current_idx + num_timesteps, trial))
            current_idx += num_timesteps
        
        # Register close function to ensure the file is closed at exit
        atexit.register(self.close)

    def __len__(self):
        # The length is the total number of timesteps across all trials
        return self.index_map[-1][1]

    def __getitem__(self, idx):
        # Find the trial corresponding to the idx using the precomputed index map
        for start_idx, end_idx, trial in self.index_map:
            if start_idx <= idx < end_idx:
                timestep_idx = idx - start_idx  # Index relative to the start of the trial

                # Create a dictionary to store images from all enabled cameras
                camera_images = {}
                for camera in self.enabled_cameras:
                    # Load the camera data
                    img = self.h5_file[trial][camera][timestep_idx]
                    img_tensor = torch.tensor(img, dtype=torch.float32)
                    
                    # Apply appropriate transform if available
                    if img_tensor.ndim == 3:  # RGB images
                        img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
                    else:  # Depth images or grayscale
                        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension

                    if camera in self.image_processors:
                        img_tensor = self.image_processors[camera](img_tensor)['pixel_values'][0]

                    # Store the tensor under the camera name key
                    camera_images[camera] = img_tensor
                
                # Load action labels (assuming action_idx is scalar and action_value is a float)
                commands = self.h5_file[trial]['commands'][timestep_idx]
                action_idx = torch.tensor(commands[:1], dtype=torch.float32)  # which action
                action_value = torch.tensor(commands[1:], dtype=torch.float32)  # action value

                # Store camera data and labels in the item dictionary
                item = {**camera_images, 'labels': torch.cat([action_idx, action_value], dim=0)}

                # Optionally include object position data
                if self.use_obj_pos:
                    obj_pos = self.h5_file[trial]['obj_pos_data'][timestep_idx]
                    item['obj_pos'] = torch.tensor(obj_pos, dtype=torch.float32)
                
                return item

        raise IndexError(f"Index {idx} is out of bounds.")

    def close(self):
        """Ensure the HDF5 file is properly closed."""
        if self.h5_file:
            self.h5_file.close()

# Example usage:
# dataset = RobotLearningDataset(hdf5_file_path='path/to/file.h5', enabled_cameras=['cam_d405_rgb'], use_obj_pos=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
