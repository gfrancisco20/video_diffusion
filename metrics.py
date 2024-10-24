# -*- coding: utf-8 -*-
"""
Metrics to evaluate perceptual quality and physical reliability of the images
"""

import torch
import torch.nn.functional as F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as  ssim
from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure as  ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as  LPIPS
import numpy as np
from fastdtw import fastdtw
import numpy as np

def custom_euclidean(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def mse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(input, target, reduction='mean')

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    """
    Compute the PSNR between two tensors for video with channels (e.g. RGB or Grayscale).

    Args:
        input: the input video tensor with shape (batch_size, num_frames, channels, height, width)
               or (num_frames, channels, height, width).
        target: the target video tensor with shape (batch_size, num_frames, channels, height, width)
                or (num_frames, channels, height, width).
        max_val: The maximum possible value in the input tensor.

    Returns:
        Average PSNR across all frames and channels in the video as a scalar tensor.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise ValueError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    psnr_values = []
    # If the input is a batch of videos, reshape to work on individual frames and channels
    if input.ndim == 5:  # (batch_size, num_frames, channels, height, width)
        batch_size, channels, num_frames, height, width = input.shape
        for i in range(num_frames):  # Iterate over the frames
          psnr_value = 10.0 * torch.log10(max_val**2 / mse(input[:, :, i, :, :], target[:, :, i, :, :]))
          psnr_values.append(psnr_value)

    elif input.ndim == 4:  # (num_frames, channels, height, width) for a single video
        channels, num_frames, height, width = input.shape
        for i in range(num_frames):  # Iterate over the frames
          psnr_value = 10.0 * torch.log10(max_val**2 / mse(input[:, i, :, :], target[:, i, :, :]))
          psnr_values.append(psnr_value)


    # Compute the average PSNR across all frames and channels
    average_psnr = torch.stack(psnr_values).mean()

    return average_psnr.item()


def ssim_video(video1, video2, scale = True, max_val = 6099):
    """
    Compute SSIM for a batch of video sequences by averaging SSIM over frames.

    Args:
        video1 (Tensor): Video tensor of shape (batch_size, channels, frames, height, width).
        video2 (Tensor): Video tensor of shape (batch_size, channels, frames, height, width).

    Returns:
        Tensor: The average SSIM over all frames and batch.
    """
    if video1.ndim == 5:
      batch_size, channels, num_frames, height, width = video1.shape
    elif video1.ndim == 4:
      channels, num_frames, height, width = video1.shape
    ssim_values = []

    # min = torch.min(video1)
    # max = torch.max(video1)
    # video2 = torch.clamp(video2, min, max)
    # video1 = (video1 - min) / (max - min)
    # video2 = (video2 - min) / (max - min)

    if scale:
      video1 = video1 / max_val
      video2 = video2 / max_val

    video1 = torch.clamp(video1, 0, 1)
    video2 = torch.clamp(video2, 0, 1)

    # Loop over the frames in the video
    for i in range(num_frames):
        # Compute SSIM for the current frame (shape: (batch_size, channels, height, width))
        if video1.ndim == 5:
          frame_ssim = ssim(video1[:, :, i, :, :], video2[:, :, i, :, :],  data_range=1.0)
        elif video1.ndim == 4:
          frame_ssim = ssim(video1[:, i, :, :].unsqueeze(0), video2[:, i, :, :].unsqueeze(0),  data_range=1.0)
        ssim_values.append(frame_ssim)

    # Stack SSIM values and compute mean over frames
    return torch.stack(ssim_values).mean().item()
  
def ms_ssim_video(video1, video2, scale = True, max_val = 6099):
    """
    Compute SSIM for a batch of video sequences by averaging SSIM over frames.

    Args:
        video1 (Tensor): Video tensor of shape (batch_size, channels, frames, height, width).
        video2 (Tensor): Video tensor of shape (batch_size, channels, frames, height, width).

    Returns:
        Tensor: The average SSIM over all frames and batch.
    """

    if video1.ndim == 5:
      batch_size, channels, num_frames, height, width = video1.shape
    elif video1.ndim == 4:
      channels, num_frames, height, width = video1.shape
    ssim_values = []

    # min = torch.min(video1)
    # max = torch.max(video1)
    # video2 = torch.clamp(video2, min, max)
    # video1 = (video1 - min) / (max - min)
    # video2 = (video2 - min) / (max - min)

    if scale:
      video1 = video1 / max_val
      video2 = video2 / max_val

    video1 = torch.clamp(video1, 0, 1)
    video2 = torch.clamp(video2, 0, 1)

    # Loop over the frames in the video
    for i in range(num_frames):
        # Compute SSIM for the current frame (shape: (batch_size, channels, height, width))
        if video1.ndim == 5:
          frame_ssim = ms_ssim(video1[:, :, i, :, :], video2[:, :, i, :, :],
                              data_range=1.0,
                              #  normalize='relu',
                              betas= (0.0448, 0.2856, 0.3001, 0.2363),
                              kernel_size=9
                              )
        else:
          frame_ssim = ms_ssim(video1[:, i, :, :].unsqueeze(0), video2[:, i, :, :].unsqueeze(0),
                              data_range=1.0,
                              #  normalize='relu',
                              betas= (0.0448, 0.2856, 0.3001, 0.2363),
                              kernel_size=9
                              )
        ssim_values.append(frame_ssim)

    # Stack SSIM values and compute mean over frames
    return torch.stack(ssim_values).mean().item()


# Initialize LPIPS metric (using 'vgg' as default backbone, but 'alex' and 'squeeze' are also available)
lpips_metric = LPIPS(net_type='squeeze')
def lpips_video(input: torch.Tensor, target: torch.Tensor, max_val = 255) -> torch.Tensor:
    """
    Compute the LPIPS between two video tensors (supports RGB or Grayscale).

    Args:
        input: the input video tensor with shape (batch_size, num_frames, channels, height, width)
               or (num_frames, channels, height, width).
        target: the target video tensor with shape (batch_size, num_frames, channels, height, width)
                or (num_frames, channels, height, width).

    Returns:
        Average LPIPS across all frames and channels in the video as a scalar tensor.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise ValueError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    if max_val!= 255:
      input = 255 * input / max_val
      target = 255 * target / max_val

    lpips_values = []
    # If the input is a batch of videos, reshape to work on individual frames and channels
    if input.ndim == 5:  # (batch_size, num_frames, channels, height, width)
        batch_size, channels, num_frames, height, width = input.shape
        for i in range(num_frames):  # Keep batch_size, channels, height, width
          lpips_value = lpips_metric(input[:, :, i, :, :].repeat(1, 3, 1, 1), target[:, :, i, :, :].repeat(1, 3, 1, 1))
          lpips_values.append(lpips_value)

    elif input.ndim == 4:  # (num_frames, channels, height, width) for a single video
        channels, num_frames, height, width = input.shape
        for i in range(num_frames):  # Keep batch_size, channels, height, width
          lpips_value = lpips_metric(input[:, i, :, :].unsqueeze(0).repeat(1, 3, 1, 1), target[:, i, :, :].unsqueeze(0).repeat(1, 3, 1, 1))
          lpips_values.append(lpips_value)


    # Compute the average LPIPS across all frames
    average_lpips = torch.stack(lpips_values).mean()

    return average_lpips.item()

def flux_by_frame(video):
  if video.ndim == 5:
    flux = video.sum(axis=(0, 1, 3, 4))
  elif video.ndim == 4:
    flux = video.sum(axis=(0, 2, 3))
  if flux.is_cuda:
    flux = flux.cpu()
  return flux.numpy()

def mpf(flux):
  return flux.max()

def t2pf(flux):
  return 2 * (np.argmax(flux) + 1)

def fluence(flux, delta_t=2*3600):
  return delta_t*flux.sum()

def dtw(pred, target, normalized=True):
  if len(pred) == 1:
    # persistence
    pred = pred[0]*np.ones_like(target)
  distance, path = fastdtw(pred, target, dist=custom_euclidean)
  if normalized:
    distance = distance / custom_euclidean(target, np.zeros_like(target)) # normalize by targel L2 norm
  return distance

def metrics_by_sim(predictions, target):
  ssims = []
  ms_ssims = []
  lpipss = []
  psnrs = []
  dtws = []
  mpfs = []
  fluences = []
  t2pfs = []
  pv_mpfs = []
  pv_fluences = []
  pv_t2pfs = []
  target_flux = flux_by_frame(target)
  target_mpf = mpf(target_flux)
  target_t2pf = t2pf(target_flux)
  target_fluence = fluence(target_flux)
  for pred in predictions:
    psnrs.append(psnr(pred, target, max_val=6099))
    ssims.append(ssim_video(pred, target))
    ms_ssims.append(ms_ssim_video(pred, target))
    lpipss.append(lpips_video(pred, target))

    pred_flux = flux_by_frame(pred)

    mpfs.append(mpf(pred_flux))
    t2pfs.append(t2pf(pred_flux))
    fluences.append(fluence(pred_flux))
    dtws.append(dtw(pred_flux, target_flux))

    pv_mpfs.append(np.abs(mpfs[-1]-target_mpf)/target_mpf)
    pv_fluences.append(np.abs(fluences[-1]-target_fluence)/target_fluence)
    pv_t2pfs.append(np.abs(t2pfs[-1]-target_t2pf))

  return target_mpf, target_t2pf, target_fluence, np.array(psnrs), np.array(ssims), np.array(ms_ssims), np.array(lpipss), np.array(mpfs), np.array(fluences), np.array(t2pfs), np.array(pv_mpfs), np.array(pv_fluences), np.array(pv_t2pfs),  np.array(dtws)

def metrics_by_sample(target_mpf, target_t2pf, target_fluence, psnrs, ssims, ms_ssims, lpipss, mpfs, fluences, t2pfs, pv_mpfs, pv_fluences, pv_t2pfs,  dtws,
                      CIs = [10,20,30,40,50,60,70,80,90,96]):
  results = {}
  results['psnr'] = psnrs.mean()
  results['psnr_std'] = psnrs.std()
  results['ssim'] = ssims.mean()
  results['ssim_std'] = ssims.std()
  results['ms_ssim'] = ms_ssims.mean()
  results['ms_ssim_std'] = ms_ssims.std()
  results['lpips'] = lpipss.mean()
  results['lpips_std'] = lpipss.std()
  results['dtws'] = dtws.mean()
  results['dtws_std'] = dtws.std()
  results['maep_mpf'] = pv_mpfs.mean()
  results['maep_mpf_std'] = pv_mpfs.std()
  results['maep_fluence'] = pv_fluences.mean()
  results['maep_fluence_std'] = pv_fluences.std()
  results['mae_t2pf_h'] = pv_t2pfs.mean()
  results['mae_t2pf_h_std'] = pv_t2pfs.std()

  target = {'mpf': target_mpf, 'fluence': target_fluence, 't2pf': target_t2pf}
  preds = {'mpf': mpfs, 'fluence': fluences, 't2pf': t2pfs}
  for ci in CIs:
    percentile_sup = (100 + ci) / 2
    percentile_inf = (100 - ci) / 2

    for mtc_name in target.keys():
      mtc = target[mtc_name]
      results[f'{mtc_name}_{ci}_inf'] = np.percentile(preds[mtc_name], percentile_inf)
      results[f'{mtc_name}_{ci}_sup'] = np.percentile(preds[mtc_name], percentile_sup)
      results[f'{mtc_name}_in_{ci}'] = (mtc >= results[f'{mtc_name}_{ci}_inf']) & (mtc <= results[f'{mtc_name}_{ci}_sup'])
      if mtc_name == 't2pf':
        results[f'{mtc_name}_{ci}_hUnc'] = (results[f'{mtc_name}_{ci}_sup'] - results[f'{mtc_name}_{ci}_inf']) / 2
      else:
        results[f'{mtc_name}_{ci}_relUnc'] = ((results[f'{mtc_name}_{ci}_sup'] - results[f'{mtc_name}_{ci}_inf']) / mtc) / 2
  return results


