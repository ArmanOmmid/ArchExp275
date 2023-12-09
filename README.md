# CSE 291B Project
# Hierarchical U-Net Vision Transformers with Residual Cross Attention for Latent Diffusion

### Setup

Download the Repository

```bash
git clone https://github.com/ArmanOmmid/XSwinDiffusion.git
```

### Install Requirements
```bash
pip install timm
pip install diffusers
pip install accelerate
pip install torchinfo
```

### Select A Model to Train On

### Custom Models
- XSwin
  - Location: /models/xswin.py
  - Relevant Custom Modules: /models/modules/normal
  - Notes: Our Segmentation Backbone

- XSwinDiffusion
  - Location: /models/xswin_diffusion.py
  - Relevant Custom Modules: /models/modules/modulated
  - Notes: Our Conditioning Modulated Denoising Backbone

### Baseline Models
- DiT
  - Location: /models/dit.py
  - Relevant Modules: None
  - Notes: The original DiT Implementation to compare with

- UViT
  - Location: /models/uvit.py
  - Relevant Modules: None
  - Notes: *Actually a DiT with UViT based skip connections. There are subtle differences.
