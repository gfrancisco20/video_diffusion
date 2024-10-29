# DDPM implementations for videos problems 

Example of probabilistic video forecast for solar corona observations (94Å): 
![Alt Text](https://github.com/gfrancisco20/video_diffusion/blob/master/Simulation_example.gif)

This repository implements the DDPM for video data of shape [channel, time, height, width] with the following variations.    
  
[diffusion/Diffusion](https://github.com/gfrancisco20/video_diffusion/blob/master/diffusion.py) implements:     
- vanilla DDPM with beta-linear schedule, [Ho et al. (2020)](https://doi.org/10.48550/arXiv.2006.11239)
- DDPM with beta-cosine schedule, [Nichol and Dhariwal (2021)](https://doi.org/10.48550/arXiv.2102.09672)
- DDIM sampling, [Song et al. (2020)](https://doi.org/10.48550/arXiv.2010.02502)
- Fast-DDPM diffusion, for improved DDIM results with reduce training time, [Jiang et al. 2024](https://doi.org/10.48550/arXiv.2405.14802)
     
[diffusion/VDiffusion](https://github.com/gfrancisco20/video_diffusion/blob/master/diffusion.py) implements:      
- Rescaled flawless linear schedule ([Lin et al., 2023]( https://doi.org/10.48550/arXiv.2305.08891)) based on v-predictions ([Salimans and Ho, 2022](https://doi.org/10.48550/arXiv.2202.00512)) for enhanced generation of events with background characteristics that significantly deviate from the dataset's average.

Two video UNet architectures are proposed, leveraging 1D-temporal convolutions and spatio-temporal attention within the bottleneck for moderate complexity:
- [module/PaletteModelVideo](https://github.com/gfrancisco20/video_diffusion/blob/master/module.py) implements 4 feature-map resolutions
- [module/PaletteModelVideoDeep](https://github.com/gfrancisco20/video_diffusion/blob/master/module.py) implements 5 feature-map resolution for more complex problems

The [training](https://github.com/gfrancisco20/video_diffusion/blob/master/training.py), [dataloaders](https://github.com/gfrancisco20/video_diffusion/blob/master/dataloaders.py), [metrics](https://github.com/gfrancisco20/video_diffusion/blob/master/metrics.py), are usage examples corresponding to the video forecasting of the solar corona presented in the paper:      
[Generative Simulations of The Solar Corona Evolution With Denoising Diffusion : Proof of Concept](https://doi.org/10.48550/arXiv.2410.20843)    
The interactive notebook [new_predictions.ipynb](https://github.com/gfrancisco20/video_diffusion/blob/master/training.py) is an example of end-to-end pipelline to perform new prediction with the resulting model.
```
```

