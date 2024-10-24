# -*- coding: utf-8 -*-
"""
Implementation of light UNets for videos leveraging temporal convolutions and spatio-temporal self-attention
PaletteModelVideo use 4 feature-map resolution
PaletteModelVideoDeep use 5 feature-map resolution, suited for bigger images and more complex problems

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class EMA:
    """
    EMA (Exponential Moving Average) model weights.
    Used to maintain a smoother version of model parameters for improved stability in generative models.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    """
    Self-attention module for 2D spatial attention. 
    Pre Layer norm  -> multi-headed tension -> skip connections -> pass it to
    the feed forward layer (layer-norm -> 2 multiheadattention)
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        channels = int(channels)
        self.channels = channels
        self.size = int(size)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class SelfAttention3D(nn.Module):
    """
    Self-attention module for 3D spatio-temporal attention. 
    Pre Layer norm  -> multi-headed tension -> skip connections -> pass it to
    the feed forward layer (layer-norm -> 2 multiheadattention)
    """
    def __init__(self, channels, size, frames):
        super(SelfAttention3D, self).__init__()
        channels = int(channels)
        self.channels = channels
        self.size = int(size)
        self.frames = frames
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size * self.frames).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.frames, self.size, self.size)


class DoubleConv(nn.Module):
    """
    2D-spatial convolution blocks for video data
    Normal convolution block, with spatial-conv  -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        mid_channels = int(mid_channels)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class DoubleConv2halfD(nn.Module):
    """
    Separated (2.5D) spatial-temporal convolution blocks for video data
    Normal convolution block, with 2d spatial-conv -> 1d temporal-conv -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        mid_channels = int(mid_channels)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=(3,1,1), padding=(1,0,0), bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0), bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class DoubleConv3D(nn.Module):
    """
    3D spatial-temporal convolution blocks for video data
    Normal convolution block, with 3d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        mid_channels = int(mid_channels)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    2D convolution-downsampling block
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer

    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                int(emb_dim),
                int(out_channels)
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1]) # projection
        return x + emb

class Down2halfD(nn.Module):
    """
    2.5D convolution-downsampling block
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer

    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            DoubleConv2halfD(in_channels, in_channels, residual=True),
            DoubleConv2halfD(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                int(emb_dim),
                int(out_channels)
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1]) # projection
        return x + emb

class Up2halfD(nn.Module):
    """
    2.5D convolution-upsampling block
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv2halfD(in_channels, in_channels, residual=True),
            DoubleConv2halfD(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(emb_dim),
                int(out_channels)
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    2D convolution-upsampling block
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(emb_dim),
                int(out_channels)
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb

class Up3D(nn.Module):
    """
    3D convolution-upsampling block
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv3D(in_channels, in_channels, residual=True),
            DoubleConv3D(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(emb_dim),
                int(out_channels)
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb

class UpSkip3D(nn.Module):
    """
    2D convolution-upsampling block appling an extra 3D convolution block to the skip connection
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.skip_conv3D = nn.Sequential(
            DoubleConv3D(in_channels//2, in_channels//2),
            DoubleConv3D(in_channels//2, in_channels//2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(emb_dim),
                int(out_channels)
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        skip_x = self.skip_conv3D(skip_x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class PaletteModelVideo(nn.Module):
    """
    PaletteModelVideo is a UNet-based architecture for video data using temporal convolutions and spatio-temporal self-attention.
    This model uses 4 feature-map resolution levels and is suitable for moderately complex image generation tasks.
    """
    def __init__(self, 
                 c_in=4, 
                 c_out=1, 
                 image_size=128, 
                 time_dim=256, 
                 device='cuda', 
                 latent=False, 
                 num_classes=None, 
                 frames = 6, 
                 bottleneck_3D = False, 
                 small=False, 
                 extra_att=False):
        """
        Initializes the PaletteModelVideo.
        Args:
            c_in (int): Number of input channels (e.g., concatenated image and auxiliary data).
            c_out (int): Number of output channels (e.g., 1  for rayscale or 3 for RGB).
            image_size (int): The size of the input images (assumed square).
            time_dim (int): Dimensionality of the time encoding.
            device (str): Device to run the model ('cuda' or 'cpu').
            latent (bool): Whether to include a latent representation in the model (default: False).
            num_classes (int): Number of classes for conditional generation (optional).
            frames (int): Number of frames for the video data.
            bottleneck_3D (bool): Whether to use 3D convolutions in the bottleneck (default: False ; which use 2.5D conv instead like for the rest of the model).
            small (bool): If True, use a reduced number of channels (default: False).
            extra_att (bool): If True, includes additional attention layers at higher resolutions than the bottleneck (default: False).
        """
        super(PaletteModelVideo, self).__init__()

        if small:
          self.chanel_factor = 2
        else:
          self.chanel_factor = 1

        # Encoder
        self.extra_att = extra_att
        self.bottleneck_3D = bottleneck_3D
        self.frames = frames
        self.true_img_size = image_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device

        starting_channels = self.image_size/self.chanel_factor
        down_channels_1   = self.image_size*2/self.chanel_factor
        down_channels_2   = self.image_size*2/self.chanel_factor
        down_channels_3   = self.image_size*4/self.chanel_factor

        bottleneck_channels = self.image_size*8/self.chanel_factor

        self.inc = DoubleConv(c_in, starting_channels) # only parallel 2D cconvolution in the first step

        self.down1 = Down2halfD(starting_channels, down_channels_1)
        # if self.extra_att:
        #   self.sa1 = SelfAttention3D(down_channels_1,int( self.true_img_size/2), frames) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down2halfD(down_channels_1, down_channels_2)
        # if self.extra_att:
        #   self.sa2 = SelfAttention3D(down_channels_2, int(self.true_img_size/4), frames)
        self.down3 = Down2halfD(down_channels_2, down_channels_3)
        if self.extra_att:
          self.sa3 = SelfAttention3D(down_channels_3, int(self.true_img_size/8), frames)

        # Bootleneck
        if self.bottleneck_3D:
          self.bot1 = DoubleConv3D(down_channels_3, bottleneck_channels)
          self.bot2 = DoubleConv3D(bottleneck_channels, bottleneck_channels)
          self.bot3 = DoubleConv3D(bottleneck_channels, down_channels_2)
        else:
          self.bot1 = DoubleConv2halfD(down_channels_3, bottleneck_channels)
          self.bot2 = DoubleConv2halfD(bottleneck_channels, bottleneck_channels)
          self.bot3 = DoubleConv2halfD(bottleneck_channels, down_channels_2)
        self.bot1att = SelfAttention3D(bottleneck_channels, self.true_img_size/8, frames) # channels, size, frames
        
        self.bot2att = SelfAttention3D(bottleneck_channels, self.true_img_size/8, frames)
        self.bot3att = SelfAttention3D(down_channels_2, self.true_img_size/8, frames)

        # Decoder: reverse of encoder
        self.up1 = Up2halfD(down_channels_2 * 2, down_channels_1)
        # if self.extra_att:
        #   self.sa4 = SelfAttention3D(down_channels_1, int(self.true_img_size/4), frames)
        self.up2 = Up2halfD(down_channels_1 * 2, starting_channels)
        # if self.extra_att:
        #   self.sa5 = SelfAttention3D(starting_channels, int(self.true_img_size/2), frames)
        self.up3 = Up2halfD(starting_channels * 2, starting_channels)
        # if self.extra_att:
        #   self.sa6 = SelfAttention3D(starting_channels, self.true_img_size, frames)
        self.outc = nn.Conv3d(int(starting_channels), c_out, kernel_size=1) # projecting back to the output channel dimensions

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # if latent == True:
        #     self.latent = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(0.2),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(0.2),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(0.2),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Flatten(),
        #         nn.Linear(64 * 8 * 8, 256)).to(device)

    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, y, lab, t):
        # Pass the source image through the encoder network
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode


        if lab is not None:
            t += self.label_emb(lab)

        # t += self.latent(y)

        # Concatenate the source image and reference image
        x = torch.cat([x, y], dim=1)

        x1 = self.inc(x) # 1
        x2 = self.down1(x1, t) # 2
        # if self.extra_att:
        #   x2 = self.sa1(x2)
        x3 = self.down2(x2, t) # 4
        # if self.extra_att:
        #   x3 = self.sa2(x3)
        x4 = self.down3(x3, t) # 4
        if self.extra_att:
          x4 = self.sa3(x4)

        x4 = self.bot1(x4) # 8
        x4 = self.bot1att(x4)
        x4 = self.bot2(x4) # 8
        x4 = self.bot2att(x4)
        x4 = self.bot3(x4) # 4
        x4 = self.bot3att(x4)

        x = self.up1(x4, x3, t) # 2 # We note that upsampling box that in the skip connections from encoder
        # if self.extra_att:
        #   x = self.sa4(x)
        x = self.up2(x, x2, t) # 1
        # if self.extra_att:
        #   x = self.sa5(x)
        x = self.up3(x, x1, t) # 1
        # if self.extra_att:
        #   x = self.sa6(x)
        output = self.outc(x)

        return output
      
class PaletteModelVideoDeep(nn.Module):
    def __init__(self, 
                 c_in=4, 
                 c_out=1, 
                 image_size=256, 
                 time_dim=256, 
                 device='cuda', 
                 latent=False, 
                 num_classes=None, 
                 frames = 6, 
                 bottleneck_3D = False, 
                 small=False, 
                 extra_att=False):    
        """
        Initializes the PaletteModelVideoDeep.
        Args:
            c_in (int): Number of input channels (e.g., concatenated image and auxiliary data).
            c_out (int): Number of output channels (e.g., 1  for rayscale or 3 for RGB).
            image_size (int): The size of the input images (assumed square).
            time_dim (int): Dimensionality of the time encoding.
            device (str): Device to run the model ('cuda' or 'cpu').
            latent (bool): Whether to include a latent representation in the model (default: False).
            num_classes (int): Number of classes for conditional generation (optional).
            frames (int): Number of frames for the video data.
            bottleneck_3D (bool): Whether to use 3D convolutions in the bottleneck (default: False ; which use 2.5D conv instead like for the rest of the model).
            small (bool): If True, use a reduced number of channels (default: False).
            extra_att (bool): If True, includes additional attention layers at higher resolutions than the bottleneck (default: False).
        """
        super(PaletteModelVideoDeep, self).__init__()

        if small:
          self.chanel_factor = 2
        else:
          self.chanel_factor = 1

        # Encoder
        self.extra_att = extra_att
        self.bottleneck_3D = bottleneck_3D
        self.frames = frames
        self.true_img_size = image_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device

        starting_channels = self.image_size/self.chanel_factor
        down_channels_1   = self.image_size*2/self.chanel_factor
        down_channels_2   = self.image_size*2/self.chanel_factor
        down_channels_3   = self.image_size*4/self.chanel_factor
        down_channels_4   = self.image_size*4/self.chanel_factor

        
        bottleneck_channels = self.image_size*8/self.chanel_factor

        self.inc = DoubleConv(c_in, starting_channels) # only parallel 2D cconvolution in the first step
        # res for 128 inputs : 128

        self.down1 = Down2halfD(starting_channels, down_channels_1) 
        # res for 128 inputs : 64 (/2)
        # if self.extra_att:
        #   self.sa1 = SelfAttention3D(down_channels_1,int( self.true_img_size/2), frames) 
        self.down2 = Down2halfD(down_channels_1, down_channels_2)
        # res for 128 inputs : 32 (/4)
        # if self.extra_att:
        #   self.sa2 = SelfAttention3D(down_channels_2, int(self.true_img_size/4), frames)
        self.down3 = Down2halfD(down_channels_2, down_channels_3)
        # res for 128 inputs : 16 (/8)
        if self.extra_att:
          self.sa_d3 = SelfAttention3D(down_channels_3, int(self.true_img_size/8), frames)
        self.down4 = Down2halfD(down_channels_3, down_channels_4)
        # res for 128 inputs : 8 (/16)
        if self.extra_att:
          self.sa_d4 = SelfAttention3D(down_channels_4, int(self.true_img_size/16), frames)

        # Bootleneck
        if self.bottleneck_3D:
          self.bot1 = DoubleConv3D(down_channels_4, bottleneck_channels)
          self.bot2 = DoubleConv3D(bottleneck_channels, bottleneck_channels)
          self.bot3 = DoubleConv3D(bottleneck_channels, down_channels_2)
        else:
          self.bot1 = DoubleConv2halfD(down_channels_3, bottleneck_channels)
          self.bot2 = DoubleConv2halfD(bottleneck_channels, bottleneck_channels)
          self.bot3 = DoubleConv2halfD(bottleneck_channels, down_channels_2)
        self.bot1att = SelfAttention3D(bottleneck_channels, self.true_img_size/16, frames) # channels, size, frames
        self.bot2att = SelfAttention3D(bottleneck_channels, self.true_img_size/16, frames)
        self.bot3att = SelfAttention3D(down_channels_3, self.true_img_size/16, frames)

        # Decoder: reverse of encoder
        self.up0 = Up2halfD(down_channels_3 * 2, down_channels_1)
        # res for 128 inputs : 16 (/8)
        if self.extra_att:
          self.sa_u0 = SelfAttention3D(down_channels_1, int(self.true_img_size/8), frames)
        self.up1 = Up2halfD(down_channels_2 * 2, down_channels_1)
        # if self.extra_att:
        #   self.sa_u1 = SelfAttention3D(down_channels_1, int(self.true_img_size/4), frames)
        self.up2 = Up2halfD(down_channels_1 * 2, starting_channels)
        # if self.extra_att:
        #   self.sa5 = SelfAttention3D(starting_channels, int(self.true_img_size/2), frames)
        self.up3 = Up2halfD(starting_channels * 2, starting_channels)
        # if self.extra_att:
        #   self.sa6 = SelfAttention3D(starting_channels, self.true_img_size, frames)
        self.outc = nn.Conv3d(int(starting_channels), c_out, kernel_size=1) # projecting back to the output channel dimensions

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # if latent == True:
        #     self.latent = nn.Sequential(
        #         nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(0.2),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(0.2),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU(0.2),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Flatten(),
        #         nn.Linear(64 * 8 * 8, 256)).to(device)

    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, y, lab, t):
        # Pass the source image through the encoder network
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode


        if lab is not None:
            t += self.label_emb(lab)

        # t += self.latent(y)

        # Concatenate the source image and reference image
        x = torch.cat([x, y], dim=1)

        x1 = self.inc(x) # 1
        x2 = self.down1(x1, t) # 2
        # if self.extra_att:
        #   x2 = self.sa1(x2)
        x3 = self.down2(x2, t) # 4
        # if self.extra_att:
        #   x3 = self.sa2(x3)
        x4 = self.down3(x3, t) # 4
        if self.extra_att:
          x4 = self.sa_d3(x4)
        x5 = self.down4(x4, t) # 4
        if self.extra_att:
          x5 = self.sa_d4(x5)

        x5 = self.bot1(x5) # 8
        x5 = self.bot1att(x5)
        x5 = self.bot2(x5) # 8
        x5 = self.bot2att(x5)
        x5 = self.bot3(x5) # 4
        x5 = self.bot3att(x5)

        x = self.up0(x5, x4, t)
        if self.extra_att:
          x = self.sa_u0(x)
        x = self.up1(x, x3, t) # 2 # We note that upsampling box that in the skip connections from encoder
        # if self.extra_att:
        #   x = self.sa4(x)
        x = self.up2(x, x2, t) # 1
        # if self.extra_att:
        #   x = self.sa5(x)
        x = self.up3(x, x1, t) # 1
        # if self.extra_att:
        #   x = self.sa6(x)
        output = self.outc(x)

        return output