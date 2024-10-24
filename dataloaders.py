# -*- coding: utf-8 -*-
"""
Video Pairing Dataset classes
PairedVideosDataset is the default implementation and can be used for real-time prediction (when no paired ground-truth available)
PairedVideosDatasetCached16b cache the dataset frames in the ram after normalisation
  --> provide a small training speed improvement depending on the GPU and model (e.g. with slow GPU and complex models involving a lot of attention, the main bottlenec is the model and not the dataloading)
"""

import torch
from torch.utils.data import Dataset
import datetime
from datetime import timedelta
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

mapping = {'quiet': 0, 'C': 1, 'M': 2, 'X': 3}

class NormaliseFromDataset(torch.nn.Module):
    def __init__(self, mean_ds, std_ds, mean=0, std=1, forward_video = False):
        """
        Args:
            mean_ds (float or list/array): The dataset's channel-wise mean.
            std_ds (float or list/array): The dataset's channel-wise standard deviation.
            mean (float): The target mean for normalization.
            std (float): The target standard deviation for normalization.
        """
        super().__init__()
        if isinstance(mean_ds, (list, tuple, torch.Tensor, np.ndarray)):  # Multi-channel case
            self.mean_ds = torch.tensor(mean_ds).float()
            self.std_ds = torch.tensor(std_ds).float()
        else:  # Single channel case
            self.mean_ds = torch.tensor([mean_ds]).float()  # Convert scalar to a 1-element tensor
            self.std_ds = torch.tensor([std_ds]).float()

        self.mean = mean
        self.std = std
        self.forward_video = forward_video

    def forward(self, tensor, chan_idx = None):
        """
        Normalize and scale the tensor to the target mean and std.
        Args:
            tensor (Tensor): Tensor to be normalized and scaled.

        Returns:
            Tensor: Normalized and rescaled tensor.
        """
        # Ensure the tensor is a float for correct scaling
        tensor = tensor.float()
        # Determine the number of channels
        num_channels = tensor.shape[0]  # Assuming shape [channels, height, width]
        if num_channels == 1:  # Single channel case
            # Normalize the tensor
            if chan_idx is None:
              chan_idx = 2
            tensor_normalized = (tensor - self.mean_ds[chan_idx]) / self.std_ds[chan_idx]
        else:  # Multi-channel case
            # Normalize the tensor (channel-wise using broadcasting)
            if self.forward_video:
              tensor_normalized = (tensor - self.mean_ds[:, None, None, None]) / self.std_ds[:, None, None, None]
            else:
              tensor_normalized = (tensor - self.mean_ds[:, None, None]) / self.std_ds[:, None, None]
        # Rescale to the new mean and standard deviation
        tensor_rescaled = tensor_normalized * self.std + self.mean
        # print('NORMALISE OOUTPUT SHAPE : ' , tensor_rescaled.shape)
        return tensor_rescaled

    def reverse_transform(self, tensor):
        """
        Reverse the normalization and scaling to recover the original tensor values.

        Args:
            tensor (Tensor): Tensor to be reverted to its original scale.

        Returns:
            Tensor: Tensor with original scaling.
        """
        # Ensure the tensor is a float for correct scaling
        tensor = tensor.float()

        # Determine the number of channels
        num_channels = tensor.shape[0]

        # Undo the rescaling to the original mean and std
        tensor_rescaled = (tensor - self.mean) / self.std

        if num_channels == 1:  # Single channel case
            # Revert normalization (channel-wise)
            tensor_original = tensor_rescaled * self.std_ds[0] + self.mean_ds[0]
        else:  # Multi-channel case
            # Revert normalization (channel-wise using broadcasting)
            tensor_original = tensor_rescaled * self.std_ds[:, None, None, None] + self.mean_ds[:, None, None, None]
        return tensor_original

    def reverse_transform_exctracted_chanel(self, tensor, original_index=2):
        """
        Apply the reverse transform corresponding to the single channel of index 'original_index' during normalisation
        """
        # Ensure the tensor is a float for correct scaling
        tensor = tensor.float()
        # Undo the rescaling to the original mean and std
        tensor_rescaled = (tensor - self.mean) / self.std
        tensor_original = tensor_rescaled * self.std_ds[original_index] + self.mean_ds[original_index]
        return tensor_original

class PairedVideosDataset(Dataset):
    def __init__(self, 
                 dataframe, 
                 root_dir, 
                 transform_video=None, 
                 time_interval_min = 120, 
                 num_frames = 6, 
                 wavelength = '0193x0211x0094', 
                 target_channel_index = None, 
                 resize = None, 
                 continous_labels = False, 
                 train = False,
                 normalisation = None, 
                 real_time = False, # to build dataset without ground truth when not available (real-time forecast)
                 time_path_format = '%Y%m%d_%H%M'):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe with ['HARPNUM', 'T', 'label'].
            root_dir (string): Directory with all the image data.
            transform_video (callable, optional): Optional transform to be applied on a sample (e.g., resizing).
        """
        self.time_path_format =  time_path_format
        self.real_time = real_time
        self.dataframe = dataframe.copy()
        self.root_dir = root_dir
        self.transform_video = transform_video
        self.num_frames = num_frames
        self.time_interval_min = timedelta(minutes=time_interval_min)  # 2-hour intervals
        self.target_channel_index = target_channel_index
        self.wavelength = wavelength
        self.incomplete_samples = pd.DataFrame(columns=['HARPNUM', 'T_sample', 'T_missing'])  # To store incomplete rows
        self.continous_labels = continous_labels
        if normalisation:
            self.Normalisation = normalisation
        else:
            self.Normalisation =  NormaliseFromDataset(mean_ds=(118/255, 
                                                                118/255, 
                                                                18/255), 
                                                       std_ds=(2.7/255, 
                                                               4.0/255, 
                                                               3.4/255), 
                                                       mean = 0.5, 
                                                       std =  0.25)
        if resize is not None:
          self.transform1 = transforms.Compose([
              transforms.Resize((resize, resize)),
              transforms.RandomVerticalFlip(p=1.0),  # flipping to align with helioviewer display
          ])
        else:
          self.transform1 = transforms.Compose([
              transforms.RandomVerticalFlip(p=1.0),
          ])
        self._validate_dataframe()

    def _validate_dataframe(self):
        """
        Iterate over the dataframe to check if all image paths for each row exist.
        If an image is missing, the row is removed and added to incomplete_samples.
        """
        valid_rows = []
        for idx, row in self.dataframe.iterrows():
            arNumber = row['HARPNUM']
            timestamp_str = row['T']
            timestamp = pd.to_datetime(timestamp_str)# .strptime( , '%Y-%m-%d %H:%M:%S')

            missing_files = False
            missing_timestamp = None

            # Check all the input paths (6 past frames)
            for i in range(self.num_frames):
                time_step = timestamp - (self.num_frames - i - 1) * self.time_interval_min
                img_path = self._get_image_path(arNumber, time_step)

                if not os.path.exists(img_path):
                    if idx % 10000 == 0:
                        print('Missing file : ', img_path)
                    missing_files = True
                    missing_timestamp = time_step.strftime('%Y%m%d_%H%M')
                    break  # If one file is missing, we skip the rest

            # Check all the target paths (6 future frames) if the input paths are valid
            if not missing_files:
                for i in range(self.num_frames):
                    time_step = timestamp + (i + 1) * self.time_interval_min
                    img_path = self._get_image_path(arNumber, time_step)

                    if not os.path.exists(img_path):
                        if idx % 10000 == 0:
                          print('Missing file : ', img_path)
                        missing_files = True
                        missing_timestamp = time_step.strftime('%Y%m%d_%H%M')
                        break  # If one file is missing, we skip the rest

            # Add valid row or move to incomplete_samples
            if not missing_files:
                valid_rows.append(row)  # Add the valid row to the new list
            else:
                # Store the missing information
                self.incomplete_samples = pd.concat([self.incomplete_samples,pd.DataFrame({
                    'HARPNUM': [arNumber],
                    'T_sample': [timestamp_str],
                    'T_missing': [missing_timestamp]
                })], ignore_index=True).reset_index(drop=True)
        self.dataframe = self.dataframe.drop_duplicates(subset='id').reset_index(drop=True)
        if 'label' not in self.dataframe.columns:
          def computeLabel(tt2c, tt2m, tt2x):
            label = 0
            if ~np.isnan(tt2c):
              if tt2c < 12:
                label = 1
            if ~np.isnan(tt2m):
              if tt2m < 12:
                label = 2
            if ~np.isnan(tt2x):
              if tt2x < 12:
                label = 3
            return label
          self.dataframe['label'] = self.dataframe.apply(lambda row: computeLabel(row['nextC_t2f_h'], row['nextM_t2f_h'], row['nextX_t2f_h']), axis=1)
        print('INCOMPLETE SAMPLES : ', len(self.incomplete_samples))
        print('COMPLETE SAMPLES   : ', len(self.dataframe))

    def _get_image_path(self, arNumber, timestamp):
        """Helper function to build the image path based on arNumber and timestamp."""
        return os.path.join(self.root_dir, f"{self.wavelength}", f"{arNumber}", f"{arNumber}_{timestamp.strftime(self.time_path_format)}_{self.wavelength}.png")

    def _load_images(self, arNumber, start_time, direction):
        """
        Load a sequence of 6 images, either in the past (input) or future (target).

        Args:
            arNumber (str): The arNumber folder to look for.
            start_time (datetime): The timestamp to start with.
            direction (str): 'input' for past images, 'target' for future images.

        Returns:
            Tensor of shape (channels, frames, height, width).
        """
        images = []
        for i in range(self.num_frames):
            if direction == 'input':
                time_step = start_time - (self.num_frames - i - 1) * self.time_interval_min  # Get past frames
            else:
                time_step = start_time + (i + 1) * self.time_interval_min  # Get future frames

            img_path = self._get_image_path(arNumber, time_step)

            if os.path.exists(img_path):
                image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
                if direction == 'target' and self.target_channel_index is not None:
                  image = image[self.target_channel_index:self.target_channel_index+1, :, :]
                image = self.transform1(image)
                if direction == 'input':
                  image = self.Normalisation(image.to(torch.float32))
                else:
                  image = self.Normalisation(image.to(torch.float32), chan_idx = self.target_channel_index)
                images.append(image)
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        # Stack images into tensor of shape (channels=3, frames=6, height, width)
        images = torch.stack(images, dim=1)  # (channels, frames, height, width)
        if self.transform_video:
          images = self.transform_video(images)
        return images

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        arNumber = row['HARPNUM']
        timestamp_str = row['T']
        label = row['label']

        # Convert timestamp string to a datetime object
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

        # Load video_input (past 6 frames) and video_target (next 6 frames)
        video_input = self._load_images(arNumber, timestamp, direction='input')
        if self.real_time:
          # no target in real time prediction
          video_target = torch.tensor(0) 
        else:
          video_target = self._load_images(arNumber, timestamp, direction='target')
        if self.continous_labels:
          label = float(label)
        return video_input, video_target, label, timestamp_str

class PairedVideosDatasetCached16b(Dataset):
    def __init__(self, dataframe, root_dir, transform_video=None, time_interval_min = 120, num_frames = 6, wavelength = '0193x0211x0094', target_channel_index = None, resize = None, continous_labels = False, train = False,normalisation = None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe with ['HARPNUM', 'T', 'label'].
            root_dir (string): Directory with all the image data.
            transform_video (callable, optional): Optional transform to be applied on a sample (e.g., resizing).
        """

        self.img_count = 0
        self.images = []
        self.labels = []
        self.timesteps = []

        self.dataframe = dataframe.copy()

        self.frame_mapping = None
        self.id2idx = {}

        self.root_dir = root_dir
        self.transform_video = transform_video
        self.num_frames = num_frames
        self.time_interval_min = timedelta(minutes=time_interval_min)  # 2-hour intervals
        self.target_channel_index = target_channel_index
        self.wavelength = wavelength
        self.incomplete_samples = pd.DataFrame(columns=['HARPNUM', 'T_sample', 'T_missing'])  # To store incomplete rows
        self.continous_labels = continous_labels


        if normalisation:
            self.Normalisation = normalisation
        else:
            self.Normalisation =  NormaliseFromDataset(mean_ds=[118.0/255, 
                                                                118.0/255, 
                                                                18.0/255], 
                                                       std_ds=[2.7/255, 
                                                               4.0/255, 
                                                               3.4/255], 
                                                       mean = 0.5, 
                                                       std =  0.25, 
                                                       forward_video = False)
        if resize is not None:
          self.transform1 = transforms.Compose([
              transforms.Resize((resize, resize)),
              transforms.RandomVerticalFlip(p=1.0),  
          ])
        else:
          self.transform1 = transforms.Compose([
              transforms.RandomVerticalFlip(p=1.0),
          ])
        self._validate_dataframe()
        print('Caching to ram...')
        self.__cache__()


    def _validate_dataframe(self):
        """
        Iterate over the dataframe to check if all image paths for each row exist.
        If an image is missing, the row is removed and added to incomplete_samples.
        """
        valid_rows = []
        for idx, row in self.dataframe.iterrows():
            arNumber = row['HARPNUM']
            timestamp_str = row['T']
            timestamp = pd.to_datetime(timestamp_str)# .strptime( , '%Y-%m-%d %H:%M:%S')

            missing_files = False
            missing_timestamp = None
            # Check all the input paths (6 past frames)
            for i in range(self.num_frames):
                time_step = timestamp - (self.num_frames - i - 1) * self.time_interval_min
                img_path = self._get_image_path(arNumber, time_step)

                if not os.path.exists(img_path):
                    if idx % 10000 == 0:
                        print('Missing file : ', img_path)
                    missing_files = True
                    missing_timestamp = time_step.strftime('%Y%m%d_%H%M')
                    break  # If one file is missing, we skip the rest

            # Check all the target paths (6 future frames) if the input paths are valid
            if not missing_files:
                for i in range(self.num_frames):
                    time_step = timestamp + (i + 1) * self.time_interval_min
                    img_path = self._get_image_path(arNumber, time_step)

                    if not os.path.exists(img_path):
                        if idx % 10000 == 0:
                          print('Missing file : ', img_path)
                        missing_files = True
                        missing_timestamp = time_step.strftime('%Y%m%d_%H%M')
                        break  # If one file is missing, we skip the rest

            # Add valid row or move to incomplete_samples
            if not missing_files:
                valid_rows.append(row)  # Add the valid row to the new list
            else:
                # Store the missing information
                self.incomplete_samples = pd.concat([self.incomplete_samples,pd.DataFrame({
                    'HARPNUM': [arNumber],
                    'T_sample': [timestamp_str],
                    'T_missing': [missing_timestamp]
                })], ignore_index=True).reset_index(drop=True)
        # Update the dataframe to only include valid rows
        self.dataframe = pd.DataFrame(valid_rows).reset_index(drop=True)
        if 'label' not in self.dataframe.columns:
          def computeLabel(tt2c, tt2m, tt2x):
            label = 0
            if ~np.isnan(tt2c):
              if tt2c < 12:
                label = 1
            if ~np.isnan(tt2m):
              if tt2m < 12:
                label = 2
            if ~np.isnan(tt2x):
              if tt2x < 12:
                label = 3
            return label
          self.dataframe['label'] = self.dataframe.apply(lambda row: computeLabel(row['nextC_t2f_h'], row['nextM_t2f_h'], row['nextX_t2f_h']), axis=1)
        self.dataframe = self.dataframe.drop_duplicates(subset='id').reset_index(drop=True)
        print('INCOMPLETE SAMPLES : ', len(self.incomplete_samples))
        print('COMPLETE SAMPLES   : ', len(self.dataframe))
        self.frame_mapping = self.dataframe[['id',	'HARPNUM',	'T']].copy().set_index('id')
        zeros = np.zeros(len(self.frame_mapping),dtype='int')
        for i in range(self.num_frames):
          self.frame_mapping[f'input_{i}'] = zeros
        for i in range(self.num_frames):
          self.frame_mapping[f'target_{i}'] = zeros

    def _get_image_path(self, arNumber, timestamp):
        """Helper function to build the image path based on arNumber and timestamp."""
        return os.path.join(self.root_dir, f"{self.wavelength}", f"{arNumber}", f"{arNumber}_{timestamp.strftime('%Y%m%d_%H%M')}_{self.wavelength}.png")

    def _load_images(self, arNumber, start_time, direction, id):
        """
        Load a sequence of 6 images, either in the past (input) or future (target).

        Args:
            arNumber (str): The arNumber folder to look for.
            start_time (datetime): The timestamp to start with.
            direction (str): 'input' for past images, 'target' for future images.

        Returns:
            Tensor of shape (channels, frames, height, width).
        """
        for i in range(self.num_frames):
            if direction == 'input':
                time_step = start_time - (self.num_frames - i - 1) * self.time_interval_min  # Get past frames
            else:
                time_step = start_time + (i + 1) * self.time_interval_min  # Get future frames
            img_path = self._get_image_path(arNumber, time_step)
            img_id = f"{arNumber}_{time_step.strftime('%Y%m%d_%H%M')}"
            if img_id not in self.id2idx:
              if os.path.exists(img_path):
                  image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
                  image = self.transform1(image)
                  # image = image.clamp(0, 255).to(torch.uint8)
                  image = self.Normalisation(image).to(torch.float16)
                  self.images.append(image)
                  self.id2idx[img_id] = self.img_count
                  self.img_count+=1
              else:
                  raise FileNotFoundError(f"Image not found: {img_path}")
            self.frame_mapping.loc[id, f'{direction}_{i}'] = int(self.id2idx[img_id])

    def __len__(self):
        return len(self.dataframe)
      
    def __cache__(self):
      # fp16 caching
      for idx in tqdm(range(len(self.dataframe))):
        self.__cacheitem__(idx)
      self.images = torch.stack(self.images, dim=0)
      self.frame_mapping = self.frame_mapping.reset_index(drop=False).drop_duplicates(subset='id').set_index('id')

    def __getitem__(self, idx):
        # fp32 cast back
        row = self.dataframe.iloc[idx]
        id = row['id']
        # img_id = f"{arNumber}_{time_step.strftime('%Y%m%d_%H%M')}"
        inputs = []
        targets = []
        for i in range(self.num_frames):
          inputs.append(self.images[self.frame_mapping.loc[id, f'input_{i}']])
          if self.target_channel_index:
            targets.append(self.images[self.frame_mapping.loc[id, f'target_{i}']][self.target_channel_index:self.target_channel_index+1, :, :])
          else:
            targets.append(self.images[self.frame_mapping.loc[id, f'target_{i}']])

        inputs = torch.stack(inputs, dim=1)
        targets = torch.stack(targets, dim=1)
        return inputs.to(torch.float32), targets.to(torch.float32), row['label'], id

    def __cacheitem__(self, idx):
        row = self.dataframe.iloc[idx]
        arNumber = row['HARPNUM']
        timestamp_str = row['T']
        # Convert timestamp string to a datetime object
        timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

        # Load video_input (past 6 frames) and video_target (next 6 frames)
        self._load_images(arNumber, timestamp, 'input',row['id'])
        self._load_images(arNumber, timestamp, 'target',row['id'])
