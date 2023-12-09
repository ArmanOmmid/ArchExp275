# CSE 291B Project
# Hierarchical U-Net Vision Transformers with Residual Cross Attention for Latent Diffusion

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/ArmanOmmid/XSwinDiffusion.git
cd DiT
```

#### Requirements
```bash
pip install timm
pip install diffusers
pip install accelerate
pip install torchinfo
```

### Custom Models
- XSwin
  - Location: /models/xswin.py
  - Relevant Custom Modules: /models/modules/normal
  - Notes: 

- XSwinDiffusion
  - Location: /models/xswin_diffusion.py
  - Relevant Custom Modules: /models/modules/normal
  - Notes: 

### Baseline Models
- DiT
  - Location: /models/dit.py
  - Relevant Modules: None
  - Notes: 

- UViT
  - Location: /models/uvit.py
  - Relevant Modules: None
  - Notes: 
