# CSE 291B Project
# Hierarchical U-Net Vision Transformers with Residual Cross Attention for Latent Diffusion

## Setup

### Download the Repository

```bash
git clone https://github.com/ArmanOmmid/XSwinDiffusion.git
```

### Install Requirements
```bash
pip install timm
pip install diffusers
pip install accelerate
pip install torchinfo
pip install --user scipy==1.11.1
```

## How to Run our Code:

### Diffusion Training
[Diffusion Training Notebook](https://colab.research.google.com/drive/1DlS1C7BBMLJaIH7sshJ-3_CfDXneiI2x?usp=sharing)

### Loss Comparison
[Loss Comparison Notebook](https://colab.research.google.com/drive/1sqc2oGJ9_-K06B3DT1Bt3Z8-7fT0J-x_?usp=sharing)

### Evaluation Metrics
[Evluation Metrics Notebook](https://colab.research.google.com/drive/1TCTImMfSzC8sv7HxgZj-aaIEqfDryxWG?usp=sharing#scrollTo=ZbNCByBP_9XS)

### Custom Models
- XSwin
  - Location: ```/models/xswin.py```
  - Relevant Custom Modules: ```/models/modules/normal```
  - Notes: Our Segmentation Backbone

- XSwinDiffusion
  - Location: ```/models/xswin_diffusion.py```
  - Relevant Custom Modules: ```/models/modules/modulated```
  - Run Script: ```run_diffusion.py```
  - Notes: Our Conditioning Modulated Denoising Backbone

### Baseline Models
- DiT
  - Location: ```/models/dit.py```
  - Relevant Modules: None
  - Run Script: ```run_diffusion_dit.py```
  - Notes: The original DiT Implementation to compare with

- UViT
  - Location: ```/models/uvit.py```
  - Relevant Modules: None
  - Run Script: ```run_diffusion_uvit.py```
  - Notes: *Actually a DiT with UViT based skip connections. There are subtle differences.
