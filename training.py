# -*- coding: utf-8 -*-
"""
DDPM training loop
"""


import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
import argparse
import shutil
from dataloaders import PairedVideosDatasetCached16b
# from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch import optim
import copy
import logging
import torch.nn as nn
# from utils import plot_images, save_images, get_data
from modules import PaletteModelVideo, EMA
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc  # Import Python's garbage collector
from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure as  ms_ssim

from utils import unzip_ds, get_anomalies_dates, count_parameters, optimizer_with_decay, save_and_plot_v2v

# usage example
PROJECT_NAME = 'corona_v2v'
WANDB_ENTITY = 'your_wandb_entity'
PROJ_FOLDER  = f'worskspace_folder/{PROJECT_NAME}'
DS_ZIP       = 'zipped_dataset_path.zip'
DIR_IM       = 'unziped_images_destination_folder'
SAMPLES_PATH = 'samples_dataframe_path.csv'
ANOMALIES    = 'folder_path_for_ANOMALIES_dates_files'

IM_RES             = 128
MIXED_PREC         = True
ACC_STEPS          = 128
GRAD_NORM_CLIP     = None
SPLIT_DATE         = pd.to_datetime('2022-04-01')
CHRONO_SPLIT       = True
RANDSPLIT_BY_AR    = None
SAVE_CHECKPOINT    = True
DEL_PREV_CKPT      = False
NUM_WORKERS        = 12
PREFETCH_FACTOR    = 8
VAM_EXAMPLE_ID    = 477
WAVELENGTH         = '0193x0211x0094'
WAVELENGTHS        = WAVELENGTH.split('x')
QUALITYTHRESHOLD    = 1
  
df_samples  = pd.read_csv(SAMPLES_PATH)
RESULT_FOLDER   = f"{PROJ_FOLDER}/results"
MODEL_FOLDER    = f"{PROJ_FOLDER}/models"

r           = 2.4 # images arcsec rresolution
bound       = 256*r - 96*r # disk coverage booudaries
df_samples  = df_samples[np.abs(df_samples['C_HPC_Y_Prop']) < bound]

print('\nNUM SAMPLES : ',len(df_samples))
print('C events    : ',(df_samples['nextC_t2f_h']<12).sum())
print('M events    : ',(df_samples['nextM_t2f_h']<12).sum())
print('X events    : ',(df_samples['nextX_t2f_h']<12).sum())
print()


def parse_arguments():
  def_run_name           = "128_2halfD"
  def_epochs             = 200
  def_batch_size         = 8
  def_image_size         = IM_RES # resize parameter
  def_device             = "cuda"
  def_lr                 = 3e-4
  def_wd                 = 1e-5
  def_frames             = 6
  def_time_interval_min  = 120
  def_channel_target_idx = 2
  def_resume_run_id      = None
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default=def_run_name, help='Name of the run')
  parser.add_argument('--epochs', type=int, default=def_epochs, help='Number of epochs')
  parser.add_argument('--batch_size', type=int, default=def_batch_size, help='Batch size')
  parser.add_argument('--image_size', type=int, default=def_image_size, help='Image size')
  parser.add_argument('--device', type=str, default=def_device, help='Device to use')
  parser.add_argument('--lr', type=float, default=def_lr, help='Learning rate')
  parser.add_argument('--wd', type=float, default=def_wd, help='Weight decay')
  parser.add_argument('--frames', type=int, default=def_frames, help='Number of frames')
  parser.add_argument('--time_interval_min', type=int, default=def_time_interval_min, help='Time interval in minutes')
  parser.add_argument('--small', action='store_true', help='Use small model')
  parser.add_argument('--mid3D', action='store_true', help='Use 3D middle layers')
  parser.add_argument('--channel_target_idx', type=int, default=def_channel_target_idx, help='Target channel index')
  parser.add_argument('--restore', action='store_true', help='Whether to restore model')
  parser.add_argument('--full_att', action='store_true', help='Whether to use 3D attention in encoder and decoder')
  parser.add_argument('--xid', type=str, default=def_resume_run_id, help='wandb run id to resume if restore is true')
  return parser.parse_args()

try:
  dates2exclude = get_anomalies_dates(ANOMALIES, WAVELENGTHS, QUALITYTHRESHOLD )
except:
  dates2exclude = []

def setup_logging(run_name):
    """
    Setting up the folders for saving the model and the results
    """
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(MODEL_FOLDER, run_name), exist_ok=True)
    os.makedirs(os.path.join(RESULT_FOLDER, run_name), exist_ok=True)
      
# Start Training
def main(args):
  
  unzip_ds(DS_ZIP, DIR_IM)
  
  resize = None
  if args.image_size != IM_RES:
    print('RESIZING TO : ', args.image_size)
    resize = args.image_size

  if not CHRONO_SPLIT:
    val_size = 0.1  # Using 80% for training as an example
    if RANDSPLIT_BY_AR:
      print('AR SPLIT')
      arnums = df_samples['HARPNUM'].unique()
      arnums_train, arnums_val = train_test_split(arnums, test_size=val_size, random_state=49)
      train_df = df_samples[df_samples['HARPNUM'].isin(arnums_train)].reset_index(drop=True)
      val_df = df_samples[df_samples['HARPNUM'].isin(arnums_val)].reset_index(drop=True)
    else:
      print('RANDOM SPLIT')
      train_df, val_df = train_test_split(df_samples, test_size=val_size, random_state=49)
      train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
  else:
    print('CHRONO SPLIT')
    df_samples['Tdt'] = pd.to_datetime(df_samples['T'])
    train_df = df_samples[df_samples['Tdt'] < SPLIT_DATE].reset_index(drop=True)
    val_df = df_samples[df_samples['Tdt'] >= SPLIT_DATE].reset_index(drop=True)


  print('Training DS')
  train_dataset = PairedVideosDatasetCached16b(
        dataframe            = train_df,
        root_dir             = DIR_IM,
        time_interval_min    = args.time_interval_min,
        num_frames           = args.frames,
        WAVELENGTH           = '0193x0211x0094',
        target_channel_index = args.channel_target_idx,
        resize = resize,
        train = True # turn on random v-flip
        )
  print('Validation DS')
  val_dataset = PairedVideosDatasetCached16b(
        dataframe            = val_df,
        root_dir             = DIR_IM,
        time_interval_min    = args.time_interval_min,
        num_frames           = args.frames,
        WAVELENGTH           = '0193x0211x0094',
        target_channel_index = args.channel_target_idx,
        resize = resize,
        )
  train_data = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,# pin_memory set to True
                            num_workers=NUM_WORKERS,
                            prefetch_factor=PREFETCH_FACTOR,
                            drop_last=False)

  val_data = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,# pin_memory set to True
                            num_workers=NUM_WORKERS,
                            prefetch_factor=PREFETCH_FACTOR,  # pin_memory set to True
                            drop_last=False)
  print('Train loader and Valid loader are up!')
        
  setup_logging(args.name)
  device = args.device  
  
  # Diffusion instantiation
  diffusion_channels = 3
  if args.channel_target_idx is not None:
    diffusion_channels = 1
  diffusion = Diffusion(img_size=args.image_size, device=device, img_channel=diffusion_channels, num_frames=args.frames)
  
  # Autoencoder instantiation
  model =  PaletteModelVideo(
      c_in=4,
      c_out=1,
      image_size=args.image_size,
      time_dim=256,
      device='cuda',
      latent=False,
      num_classes=4,
      frames = args.frames,
      bottleneck_3D = args.mid3D,
      small=args.small,
      extra_att=args.full_att
  ).to(device)
  print('Model size : ')
  count_parameters(model)
  ema = EMA(0.995)
  ema_model = copy.deepcopy(model).eval().requires_grad_(False)
  optimizer = optimizer_with_decay(model, optim.AdamW, args.lr, args.wd)

  # Wandb setup and eventual resoring
  logger = SummaryWriter(os.path.join("runs", args.name))
  if args.resume_run_id:
    wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY, id=args.resume_run_id, resume="allow")
  else:
    wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY)
    wandb.config = {"# Epochs" : args.epochs,
                    "Batch size" : args.batch_size,
                    "Image size" : args.image_size,
                    "Device" : args.device,
                    "Lr" : args.lr,
                    "Wd" : args.wd,
                    "Frames" : args.frames,
                    "Timeres" : args.time_interval_min,
                    "Small" : args.small,
                    "Bottleneck3D" : args.mid3D,
                    "Target channel" : args.channel_target_idx,
                    "Full attention" : args.full_att,
                    }
  wandb.watch(model, log=None)
  starting_epc = 0
  if args.restore:
    model_ckpt_path =  wandb.restore(os.path.join(MODEL_FOLDER, args.name ,'model_ckpt.pt')).name
    ema_ckpt_path = wandb.restore(os.path.join(MODEL_FOLDER, args.name , 'ema_model_ckpt.pt')).name
    model_checkpoint = torch.load(model_ckpt_path)
    ema_checkpoint = torch.load(ema_ckpt_path)
    model.load_state_dict(model_checkpoint)
    ema_model.load_state_dict(ema_checkpoint['model_state'])
    try:
      optimizer.load_state_dict(ema_checkpoint['optimizer_state'])
    except Exception as e:
      print('Couldnt load optimizer state!')
      print(e)
    starting_epc = ema_checkpoint['epoch'] + 1
    print('\nResuming from epoch : ', starting_epc,'\n')

  # Loss and optimizer setup
  mse = nn.MSELoss()
  min_valid_loss = np.inf
  scaler = GradScaler()
  accumulation_steps = ACC_STEPS / args.batch_size
  if MIXED_PREC:
    print('\nMIXED PRECISION')
  if GRAD_NORM_CLIP:
    print(f'\nGRADIENT CLIPPING: {GRAD_NORM_CLIP}')
  print()

  # Training loop
  for epoch in range(starting_epc,args.epochs):
      logging.info(f"Starting epoch {epoch}:")
      pbar = tqdm(train_data)
      model.train()
      train_loss = 0.0
      # psnr_train = 0.0
      for i, (video_input, video_target, label, time) in enumerate(pbar):
          input = video_input.to(device).float()
          target = video_target.to(device).float()

          label = label
          time = time

          t = diffusion.sample_timesteps(target.shape[0]).to(device)
          x_t, noise = diffusion.noise_images(target, t)

          labels = None
          if MIXED_PREC:
            with autocast():
              predicted_noise = model(x_t, input, labels, t)
              loss = mse(noise, predicted_noise) / accumulation_steps
          else:
            predicted_noise = model(x_t, input, labels, t)
            loss = mse(noise, predicted_noise) / accumulation_steps

          # Outflows management
          # Calculate the percentage of NaN values in predicted_noise
          nan_percentage = torch.isnan(predicted_noise).float().mean().item() * 100
          # Issue a warning if NaN values are present but less than 50%
          if nan_percentage > 0 and nan_percentage <= 50:
              print(f"\nWARNING: {nan_percentage:.2f}% of predicted_noise values are NaN.")
          # Raise an exception if NaN percentage exceeds 50%
          elif nan_percentage > 50:
              raise ValueError(f"\nERROR: {nan_percentage:.2f}% of predicted_noise values are NaN. Stopping execution.")
            
          if MIXED_PREC:
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
              # Gradient clipping if enabled
              if GRAD_NORM_CLIP:
                  scaler.unscale_(optimizer) 
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_NORM_CLIP)
              scaler.step(optimizer)
              scaler.update()
              optimizer.zero_grad()
          else:
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
              if GRAD_NORM_CLIP:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_NORM_CLIP)
              optimizer.step()
              optimizer.zero_grad()

          ema.step_ema(ema_model, model)

          batch_loss = loss.detach().item() * accumulation_steps * input.size(0)
          train_loss += batch_loss
          pbar.set_postfix(loss=loss.item())
          logger.add_scalar("loss", batch_loss, global_step=epoch * len(pbar) + i)

      if (i + 1) % accumulation_steps != 0:
        if MIXED_PREC:
          if GRAD_NORM_CLIP:
              scaler.unscale_(optimizer)
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_NORM_CLIP)
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()
        else:
          if GRAD_NORM_CLIP:
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_NORM_CLIP)
          optimizer.step()
          optimizer.zero_grad()

      if epoch % 2 == 0:
        example_idx  = 0 
        ema_sampled_images = diffusion.sample(ema_model, y=input[example_idx].reshape(1, 3, args.frames, args.image_size, args.image_size), labels=None, n=1)
        v_input = val_dataset.Normalisation.reverse_transform(input[example_idx].cpu()).reshape(3, args.frames, args.image_size, args.image_size).permute(1, 2, 3, 0).cpu().numpy()
        v_target = val_dataset.Normalisation.reverse_transform_exctracted_chanel(target[example_idx].cpu()).reshape(1, args.frames, args.image_size, args.image_size).permute(1, 2, 3, 0).cpu().numpy()
        ema_samp = val_dataset.Normalisation.reverse_transform_exctracted_chanel(ema_sampled_images[example_idx].cpu()).permute(1, 2, 3, 0).cpu().numpy()
        # save and plot
        save_path_train = os.path.join(RESULT_FOLDER, args.name, f"{epoch}_ema_v2v_train.gif")
        save_and_plot_v2v(v_input, v_target, ema_samp, save_path_train, title=f'TRAINING : Time: {time[0]} -- Label: {label[0]} -- Epoch : {epoch}', fps=4)

      # Clean up memory before validation
      torch.cuda.empty_cache()
      gc.collect()

      # Validation step
      valid_loss = 0.0
      # psnr_val = 0.0
      pbar_val = tqdm(val_data)
      model.eval()
      with torch.no_grad():
          for i, (video_input, video_target, label, time) in enumerate(pbar_val):
              input = video_input.to(device).float()
              target = video_target.to(device).float()
              label = label
              time = time


              labels = None
              t = diffusion.sample_timesteps(target.shape[0]).to(device)
              x_t, noise = diffusion.noise_images(target, t)

              # with autocast():
              if MIXED_PREC:
                  with autocast():
                      predicted_noise = model(x_t, input, labels, t)
                      loss = mse(noise, predicted_noise)
              else:
                  predicted_noise = model(x_t, input, labels, t)
                  loss = mse(noise, predicted_noise)

              valid_loss += loss.detach().item() * input.size(0)
              pbar_val.set_postfix(loss=loss.item())
              logger.add_scalar("val_loss", loss.item() * input.size(0), global_step=epoch * len(pbar_val) + i)

              # Logging and saving
              mod = VAM_EXAMPLE_ID % args.batch_size
              if i > 0:
                if epoch % 2 == 0 and ((VAM_EXAMPLE_ID - mod ) // (i) == args.batch_size) and ((VAM_EXAMPLE_ID - mod ) / (i) <= args.batch_size):
                    example_idx  =  mod
                    with torch.no_grad():
                      ema_sampled_images = diffusion.sample(ema_model, y=input[example_idx].reshape(1, 3, args.frames, args.image_size, args.image_size), labels=None, n=1)
                    # save_tensor_as_gif(ema_sampled_images, os.path.join("results_256_DDPM_v2", args.name, f"{epoch}_ema_cond.gif"))
                    v_input = val_dataset.Normalisation.reverse_transform(input[example_idx].cpu()).reshape(3, args.frames, args.image_size, args.image_size).permute(1, 2, 3, 0).cpu().numpy()
                    v_target = val_dataset.Normalisation.reverse_transform_exctracted_chanel(target[example_idx].cpu()).reshape(1, args.frames, args.image_size, args.image_size).permute(1, 2, 3, 0).cpu().numpy()
                    ema_samp = val_dataset.Normalisation.reverse_transform_exctracted_chanel(ema_sampled_images[0].cpu()).permute(1, 2, 3, 0).cpu().numpy()
                    # save and plot
                    save_path_val = os.path.join(RESULT_FOLDER, args.name, f"{epoch}_ema_v2v_val.gif")
                    save_and_plot_v2v(v_input, v_target, ema_samp, save_path_val, title=f'VALIDATION : Time: {time[0]} -- Label: {label[0]} -- Epoch : {epoch}', fps=4)

      # Clean up memory after validation
      torch.cuda.empty_cache()
      gc.collect()

      # Logging and saving
      if epoch % 2 == 0:
          wandb.log({
              "Training Loss": train_loss / (len(train_data)*args.batch_size),
              "Validation Loss": valid_loss / (len(val_data)*args.batch_size),
              'Sample validation': wandb.Video(save_path_val, fps=4, format=save_path_val.split('.')[-1]),
              'Sample training': wandb.Video(save_path_train, fps=4, format=save_path_val.split('.')[-1])
          })
          plt.close()
      else:
          wandb.log({
              "Training Loss": train_loss / (len(train_data)*args.batch_size),
              "Validation Loss": valid_loss / (len(val_data)*args.batch_size),
          })

      if min_valid_loss > valid_loss:
          logging.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})) \t Saving Best Model')
          min_valid_loss = valid_loss

          torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, args.name, "model_best.pt"))
          # torch.save(ema_model.state_dict(), os.path.join("models_512_DDPM_v2", args.name, f"ema_ckpt_cond.pt"))
          state = {
              'model_state': ema_model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'epoch': epoch
          }
          torch.save(state, os.path.join(MODEL_FOLDER, args.name, "ema_model_best.pt"))

      if SAVE_CHECKPOINT:
        print('Saving checkpoint')
        # Saving State Dict
        model_ckpt_path = os.path.join(MODEL_FOLDER, args.name,'model_ckpt.pt')
        ema_ckpt_path = os.path.join(MODEL_FOLDER, args.name, 'ema_model_ckpt.pt')
        torch.save(model.state_dict(), model_ckpt_path)
        # torch.save(ema_model.state_dict(), os.path.join("models_512_DDPM_v2", args.name, f"ema_ckpt_cond.pt"))
        state = {
            'model_state': ema_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, ema_ckpt_path)
        wandb.save(model_ckpt_path)
        wandb.save(ema_ckpt_path)

if __name__ == "__main__":
    args = parse_arguments()
    # configuration used in the paper
    args.full_att = True
    args.mid3D    = False
    args.small    = True
    main(args)
