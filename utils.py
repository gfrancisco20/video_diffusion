# -*- coding: utf-8 -*-
"""
Vizualisation and training utilites
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import pandas as pd
from PIL import Image
import os
import shutil
import logging
import torch.nn as nn

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters
    non_trainable_params = total_params - trainable_params  # Non-trainable parameters

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

def optimizer_with_decay(model, Optimizer, lr, wd):
  decay = []
  no_decay = []
  # Loop through all named parameters
  for name, param in model.named_parameters():
      # Default: Apply weight decay to all parameters
      apply_decay = True

      # Loop through all named modules (layers)
      for module_name, module in model.named_modules():
          # Check if this parameter belongs to a normalization layer
          if module_name in name:
              if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                  apply_decay = False
                  break
              # Optionally, exclude biases from weight decay
              if 'bias' in name:
                  apply_decay = False
                  break
      # Add the parameter to the appropriate list
      if apply_decay:
          decay.append(param)
      else:
          no_decay.append(param)
  # Print the number of parameters without weight decay
  print(f"Number of parameters without weight decay: {sum(p.numel() for p in no_decay)}")
  print(f"Number of parameters with weight decay: {sum(p.numel() for p in decay)}")
  # Set up AdamW optimizer
  optimizer = Optimizer([
      {'params': decay, 'weight_decay'    : wd},  # Apply weight decay to other parameters
      {'params': no_decay, 'weight_decay' :  0.0}  # Exclude normalization layers and biases from weight decay
  ], lr = lr)
  return optimizer


def unzip_ds(ds_zip, dir_im, ds_zip_2_scratch = True):
  if ds_zip is not None:
    if not os.path.exists(dir_im):
      os.makedirs(dir_im,exist_ok=True)
    if ds_zip_2_scratch:
      ds_zip = shutil.copy(ds_zip, dir_im)
      shutil.unpack_archive(ds_zip, dir_im, "zip")
      os.remove(ds_zip)
    else:
      shutil.unpack_archive(ds_zip, dir_im, "zip")
  print('DS Unzipped')

  
def get_anomalies_dates(anomaliesGradeFolder, wavelengths, qualityTreshold = 1):
  # see https://zenodo.org/records/11058938 >> anomalies.zip
  anomaliesDates = {}
  for idx, w in enumerate(wavelengths):
    anomaliesDates[w] = pd.read_csv(f'{anomaliesGradeFolder}/{w}_anomalies_notes.csv')
    anomaliesDates[w]['timestamp'] = pd.to_datetime(anomaliesDates[w]['timestamp'])
    anomaliesDates[w] = anomaliesDates[w].set_index('timestamp')
    anomaliesDates[w] = anomaliesDates[w][anomaliesDates[w]['grade'] < qualityTreshold]
    if idx==0:
      dates2exclude = anomaliesDates[w].index.values
    else:
      dates2exclude = np.concatenate([dates2exclude, anomaliesDates[w].index.values])
  dates2exclude.shape, np.unique(dates2exclude).shape
  dates2exclude = np.sort(np.unique(dates2exclude))
  return dates2exclude

def descaling(scaled_image, max = 6099.48465):
  max = np.log(1+max)
  image = np.exp(max * scaled_image / 255) - 1
  return image

def save_and_plot_v2v(video_input, video_target, video_prediction, save_path=None, title="", fps=2, target_channel=2, html_plot=False):
    """
    Creates and displays an animation showing three videos side by side in a Jupyter notebook.

    Args:
        video_input: Numpy array for the true input video of shape (frames, height, width, channels)
        video_target: Numpy array for the ground truth of shape (frames, height, width, channels)
        video_prediction: Numpy array for the predicted video of shape (frames, height, width, channels)
        save_path: Where to save the video (optional, saves as GIF)
        title: Optional title for the entire video.
        fps: Frames per second for the output video.
    """
    if target_channel is not None:
        video_input = video_input[:, :, :, target_channel:target_channel+1]

    vmin=np.min([video_input.min(),video_target.min()])
    vmax=np.max([np.percentile(video_input,99.9),np.percentile(video_target,99.9)])

    # Number of frames
    frames = video_input.shape[0]
    images = []

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Set the initial frame in all subplots
    im1 = ax1.imshow(video_input[0], origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title('12 previous hours')

    im2 = ax2.imshow(video_target[0], origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title('True next 12h')

    im3 = ax3.imshow(video_prediction[0], origin='lower', vmin=vmin, vmax=vmax)
    ax3.set_title('Predicted next 12h')

    # Add a big title in the middle of all subplots
    fig.suptitle(title)

    def update(frame_idx):
        """Updates the plot for the given frame index."""
        im1.set_array(video_input[frame_idx])
        im2.set_array(video_target[frame_idx])
        im3.set_array(video_prediction[frame_idx])
        return [im1, im2, im3]

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)

    # If a save path is provided, save the animation
    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=fps)

    plt.tight_layout()

    # Display the animation in the notebook
    if html_plot:
        plt.close(fig)
        return HTML(ani.to_jshtml())
    else:
      plt.show()
      return ani

def save_and_plot_err(video_input, video_target, video_prediction, video_persistence , save_path=None, title="", fps=2, target_channel=2, html_plot=False):
    """
    Creates and displays an animation showing three videos side by side in a Jupyter notebook.

    Args:
        video_input: Numpy array for the true input video of shape (frames, height, width, channels)
        video_target: Numpy array for the ground truth of shape (frames, height, width, channels)
        video_prediction: Numpy array for the predicted video of shape (frames, height, width, channels)
        save_path: Where to save the video (optional, saves as GIF)
        title: Optional title for the entire video.
        fps: Frames per second for the output video.
    """
    if target_channel is not None:
        video_input = video_input[:, :, :, target_channel:target_channel+1]

    # Number of frames
    frames = video_input.shape[0]
    images = []

    video_prediction = video_prediction - video_target
    video_persistence = video_persistence - video_target

    rmse_pred = np.sqrt(np.square(video_prediction).mean())
    rmse_pers = np.sqrt(np.square(video_persistence).mean())

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    # Set the initial frame in all subplots
    im1 = ax1.imshow(video_input[0], origin='lower')
    ax1.set_title('12 previous hours')

    im2 = ax2.imshow(video_target[0], origin='lower')
    ax2.set_title('True next 12h')

    im3 = ax3.imshow(video_prediction[0], origin='lower')
    ax3.set_title(f'Predicted error (RMSE : {rmse_pred:.2f})')

    im4 = ax4.imshow(video_persistence[0], origin='lower')
    ax4.set_title(f'Persistence error (RMSE : {rmse_pers:.2f})')

    # Add a big title in the middle of all subplots
    fig.suptitle(title)

    def update(frame_idx):
        """Updates the plot for the given frame index."""
        im1.set_array(video_input[frame_idx])
        im2.set_array(video_target[frame_idx])
        im3.set_array(video_prediction[frame_idx])
        im4.set_array(video_persistence[frame_idx])
        return [im1, im2, im3, im4]

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)

    # If a save path is provided, save the animation
    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=fps)

    plt.tight_layout()

    if html_plot:
      plt.close(fig)
      return HTML(ani.to_jshtml())
    else:
      plt.show()
      return ani
      
  
def save_and_plot_simus(video_input, video_target, predictions, simus , save_path=None, title="", target_channel=2):
    """
    Creates and displays an animation showing three videos side by side in a Jupyter notebook.

    Args:
        video_input: Numpy array for the true input video of shape (frames, height, width, channels)
        video_target: Numpy array for the ground truth of shape (frames, height, width, channels)
        video_prediction: Numpy array for the predicted video of shape (frames, height, width, channels)
        save_path: Where to save the video (optional, saves as GIF)
        title: Optional title for the entire video.
        fps: Frames per second for the output video.
    """
    if target_channel is not None:
        video_input = video_input[:, :, :, target_channel:target_channel+1]

    std_map = np.std(np.concatenate([np.expand_dims(pred/255,axis=0) for pred in predictions],axis=0),axis=0)
    mae_map = np.mean(np.concatenate([np.expand_dims(100*(pred-video_target)/video_target,axis=0) for pred in predictions],axis=0),axis=0)

    # Create a figure with three subplots
    fig, ax = plt.subplots(len(simus) + 4, 6, figsize=(36, 6 * (len(simus) + 4)-1))

    # Set the initial frame in all subplots
    for frame_idx in range(6):

      vmin=np.min([video_input.min(),video_target.min()])
      vmax=np.max([np.percentile(video_input,99.9),np.percentile(video_target,99.9)])

      ax[0,frame_idx].imshow(video_input[frame_idx], origin='lower', vmin=vmin, vmax=vmax)
      ax[0,frame_idx].set_title(f'-{(5-frame_idx)*2}H')
      ax[0,frame_idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
      ax[0,frame_idx].spines['top'].set_visible(False)
      ax[0,frame_idx].spines['right'].set_visible(False)
      ax[0,frame_idx].spines['bottom'].set_visible(False)
      ax[0,frame_idx].spines['left'].set_visible(False)
      if frame_idx == 0:
        ax[0,frame_idx].set_ylabel('Input', rotation=0, labelpad = 90)

      ax[1,frame_idx].imshow(video_target[frame_idx], origin='lower', vmin=vmin, vmax=vmax)
      ax[1,frame_idx].set_title(f'+{(frame_idx+1)*2}H')
      ax[1,frame_idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
      ax[1,frame_idx].spines['top'].set_visible(False)
      ax[1,frame_idx].spines['right'].set_visible(False)
      ax[1,frame_idx].spines['bottom'].set_visible(False)
      ax[1,frame_idx].spines['left'].set_visible(False)
      if frame_idx == 0:
        ax[1,frame_idx].set_ylabel('Ground\nTruth', rotation=0, labelpad = 90)

      for i in range(len(simus)):
        ax[i+2,frame_idx].imshow(predictions[simus[i]][frame_idx], origin='lower', vmin=vmin, vmax=vmax)
        # ax[i+2,frame_idx].set_title(f'Simu #{i} (+{(frame_idx+1)*2}H)')
        ax[i+2,frame_idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[i+2,frame_idx].spines['top'].set_visible(False)
        ax[i+2,frame_idx].spines['right'].set_visible(False)
        ax[i+2,frame_idx].spines['bottom'].set_visible(False)
        ax[i+2,frame_idx].spines['left'].set_visible(False)
        if frame_idx == 0:
          ax[i+2,frame_idx].set_ylabel(f'Simulation\n #{i}', rotation=0, labelpad = 110)


      cmap = 'bwr'
      vmin=-np.max(np.abs(mae_map))
      vmax=np.max(np.abs(mae_map))
      im_mae=ax[len(simus)+2,frame_idx].imshow(mae_map[frame_idx], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
      # ax[len(simus)+2,frame_idx].set_title(f'20-Simus MPE-map (+{(frame_idx+1)*2}H)')
      ax[len(simus)+2,frame_idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
      ax[len(simus)+2,frame_idx].spines['top'].set_visible(False)
      ax[len(simus)+2,frame_idx].spines['right'].set_visible(False)
      ax[len(simus)+2,frame_idx].spines['bottom'].set_visible(False)
      ax[len(simus)+2,frame_idx].spines['left'].set_visible(False)
      if frame_idx == 0:
        ax[len(simus)+2,frame_idx].set_ylabel('Mean %-error\n(from 20\n simulations)', rotation=0, labelpad = 110)

      cmap = 'magma'
      vmin=np.min(std_map)
      vmax=np.max(std_map)
      im_std=ax[len(simus)+3,frame_idx].imshow(std_map[frame_idx], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
      # ax[len(simus)+3,frame_idx].set_title(f'20-Simus Std-map (+{(frame_idx+1)*2}H)')
      ax[len(simus)+3,frame_idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
      ax[len(simus)+3,frame_idx].spines['top'].set_visible(False)
      ax[len(simus)+3,frame_idx].spines['right'].set_visible(False)
      ax[len(simus)+3,frame_idx].spines['bottom'].set_visible(False)
      ax[len(simus)+3,frame_idx].spines['left'].set_visible(False)
      if frame_idx == 0:
        ax[len(simus)+3,frame_idx].set_ylabel('Std-map\n(from 20\n simulations)', rotation=0, labelpad = 110)

       # Add a colorbar only to the last column for the two last rows
      cbar_ax_mae = fig.add_axes([1.0, 0.115 + 0.04, 0.005, 0.117])  # Add an axis for the MAE colorbar
      cbar_mae=fig.colorbar(im_mae, cax=cbar_ax_mae)#, label='Mean-Dev')
      cbar_mae.ax.tick_params(labelsize=24)

      cbar_ax_std = fig.add_axes([1.0, 0.014, 0.005, 0.117])  # Add an axis for the Std colorbar
      cbar_std=fig.colorbar(im_std, cax=cbar_ax_std)#, label='Std')
      cbar_std.ax.tick_params(labelsize=24)

      # Add a big title in the middle of all subplots
      # fig.suptitle(title, y=1.01)

    # If a save path is provided, save the animation
    if save_path is not None:
        plt.save(save_path)

    plt.tight_layout()
    plt.show()
  
  
def plot_reliability_diagram(var_name, stat):
    """
    Generic function to plot reliability diagram for a given variable using data from 'stat' series.
    """
    # Extract the data from the 'stat' series
    df = extract_data_for_variable(stat, var_name)

    # Plot
    plt.figure(figsize=(8, 6))

    # Line plot for var_in
    plt.plot(df['CI'], df[f'{var_name}_in'], label=f'{var_name.upper()} (%)', marker='o', color='b')

    # Error bars for uncertainties
    plt.errorbar(df['CI'], df[f'{var_name}_in'], yerr=df[f'{var_name}_unc'], fmt='o', color='r', label=f'Average CI size\n(observed {var_name} %)')

    # Diagonal x=y line
    plt.plot(df['CI'], df['CI'], '--', color='gray', label='x = y (perfect)')

    plt.xlabel('Confidence Interval (CI)')
    plt.ylabel(f'Observed {var_name.upper()} within CI (%)')
    plt.title(f"{var_name.upper()}\'s Reliability Diagram")
    plt.legend()
    plt.grid(True)

    plt.ylim(0, 100)
    plt.xlim(0, 100)

    plt.show()

def extract_data_for_variable(stat, var_name):
    """
    Extracts the 'in' and uncertainty values for a given variable from the 'stat' series.
    """
    CI = [int(key.split('_')[-1]) for key in stat.keys() if key.startswith(f'{var_name}_in')]
    var_in = [stat[f'{var_name}_in_{ci}'] * 100 for ci in CI]

    if var_name != "t2pf":
        var_unc = [stat[f'{var_name}_{ci}_relUnc'] * 100 for ci in CI]
    else:
        var_unc = [stat[f'{var_name}_{ci}_hUnc'] for ci in CI]

    return pd.DataFrame({'CI': [0]+CI+[100], f'{var_name}_in': [0]+var_in+[100], f'{var_name}_unc': [0]+var_unc+[0]})

def plot_reliability_diagram(var_name, stat, ax1, right_y_min=None, right_y_max=None):
    """
    Generic function to plot reliability diagram for a given variable using data from 'stat' series.
    Adds a right y-axis for 't2pf' to display values in hours. The right axis limits can be set with right_y_min and right_y_max.
    """
    color_curve = 'b'
    marker_curve = 'o'
    # error bars
    elinewidth=1
    capsize=2
    markersize=4
    fmt='o'
    color='r'

    # Extract the data from the 'stat' series
    df = extract_data_for_variable(stat, var_name)

    # fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the var_in line for the left y-axis
    ax1.plot(df['CI'], df[f'{var_name}_in'], label=f'{var_name.upper()} (%)', marker=marker_curve, color=color_curve)

    # Diagonal x=y line
    ax1.plot(df['CI'], df['CI'], '--', color='gray', label='x = y (perfect)')

    # Labels for the left y-axis and x-axis
    ax1.set_xlabel('Simulations Confidence Interval (CI)')
    ax1.set_ylabel(f'Observed {var_name.upper()} within CI (%)')

    # Title
    ax1.set_title(f"{var_name.upper()}\'s Reliability Diagram")
    ax1.grid(True)

    # Plot the error bars for var_in on the appropriate axis

    if var_name == "t2pf":
        # Add right y-axis for t2pf to display values in hours
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        ax2.set_ylabel(f'Average CI size\n(in hours)', color='r')
        df[f'{var_name}_in'] = right_y_max * df[f'{var_name}_in'] / 100 + right_y_min
        ax2.errorbar(df['CI'], df[f'{var_name}_in'], yerr=df[f'{var_name}_unc'], fmt=fmt, color=color,
                     label=f'Average CI size\n(in hours)', elinewidth=elinewidth, capsize=capsize, markersize=markersize)
        # ax2.errorbar(df['CI'], np.zeros_like(df[f'{var_name}_in']), yerr=2*df[f'{var_name}_unc'], fmt=fmt, color=color,
        #              label=f'Average CI size\n(in hours)', elinewidth=elinewidth, capsize=capsize, markersize=markersize)
        ax2.tick_params(axis='y', labelcolor='r')

        ax2.legend(loc='lower right')

        ax2.set_yticklabels([r'2H' for tick in ax2.get_yticks() ], rotation=90) # $\updownarrow$
        ax2.set_yticks(ax2.get_yticks() + right_y_max/10)
        ax2.tick_params(axis='y', length=0,)
        # for tick in ax2.get_yticks():
          # ax2.annotate(r'$\updownarrow$', xy=(120, tick), xytext=(120, tick), textcoords='data', fontsize=70,
          #             ha='right', va='center', color='red')

        # Set custom limits for the right y-axis if provided
        if right_y_min is not None and right_y_max is not None:
            ax2.set_ylim(right_y_min, right_y_max)

    else:
        # Plot error bars for other variables (on the left y-axis)
        ax1.errorbar(df['CI'], df[f'{var_name}_in'], yerr=df[f'{var_name}_unc'], fmt=fmt, color=color,
                     label=f'Average CI size\n(in observed {var_name} %)', elinewidth=elinewidth, capsize=capsize, markersize=markersize)
        # ax1.errorbar(df['CI'], np.zeros_like(df[f'{var_name}_in']), yerr=2*df[f'{var_name}_unc'], fmt=fmt, color=color,
        #              label=f'Average CI size\n(in observed {var_name} %)', elinewidth=elinewidth, capsize=capsize, markersize=markersize)

    # Legend
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 100)

    # plt.show()

# Example call for "t2pf" with custom right y-axis limits
# plot_reliability_diagram("t2pf", stat, right_y_min=0, right_y_max=10)

def plot_reliability_diagrams_all(stat):
    """
    Plot reliability diagrams for 'fluence', 'mpf', and 't2pf' on a (1,3) grid.
    This uses the existing 'plot_reliability_diagram_for_grid' function to create the individual plots.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create a 1x3 grid of subplots
    variables = ['fluence', 'mpf', 't2pf']

    for i, var_name in enumerate(variables):
        # Call the plot function for each variable and assign it to the specific subplot
        if var_name == "t2pf":
            plot_reliability_diagram(var_name, stat, axs[i], right_y_min=0, right_y_max=10)
        else:
            plot_reliability_diagram(var_name, stat, axs[i])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()
    
    