# Hierarchical U-Net Vision Transformers with Residual Cross Attention for Latent Diffusion

## Authors
Arman Ommid, Mayank Jain

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

### Inference Sampling
[Inference Sampling Notebook](https://colab.research.google.com/drive/1vJE2atdZ2mLJd0d2ub7XV5-od5D_8H7H?usp=sharing)

### Evaluation Metrics
[Evaluation Metrics Notebook](https://colab.research.google.com/drive/1TCTImMfSzC8sv7HxgZj-aaIEqfDryxWG?usp=sharing#scrollTo=ZbNCByBP_9XS)

### Custom Models
- XSwin
  - **Location:** ```/models/xswin.py```
  - **Relevant Custom Modules:** ```/models/modules/normal```
  - **Description:** *Our Segmentation Backbone*
  - **Implementation:** *As an isolated segmentation network, XSwin is largely based on SwinV2 Blocks supported by outer convolutional blocks and inner global attention ViT blocks for the bottleneck. This promotes heiarchical, multiscale learning with appropriate inductive biases. The architecture also feature localized residual cross attention that dynamically aggregate shallow encoder features for refinement before being combined with deep decoder features for further processing. The ViT bottleneck recieves positional embeddings with the features while the convolutional skip connection is just traditional concatonation.*

- XSwinDiffusion
  - **Location:** ```/models/xswin_diffusion.py```
  - **Relevant Custom Modules:** ```/models/modules/modulated```
  - **Run Script:** ```run_diffusion.py```
  - **Description:** *Our Conditioning Modulated Denoising Backbone*
  - **Implementation:** *We take our XSwin isolated segmentation backbone and make the following modifications. First, we create frozen parameters for time step and class label conditioning based on the number of diffusion steps and the number of classes using Fourier based embeddings. We then augment all parameterized layers with conditioning modulation layers using adaptive layer normalization as per DiT to encode conditioning information for time steps and class labels efficiently. We also make sure that the input and output hidden dimensions are modifiable, ensuring the output dimensions output both the predicted image and the predicted noise as per the DiT diffusion pipeline. We also implement an additional forward function focused on classifier free guidance as per DiT.*

### Baseline Models
- DiT
  - **Location:** ```/models/dit.py```
  - **Relevant Modules:** None
  - **Run Script:** ```run_diffusion_dit.py```
  - **Description:** *The original DiT Implementation to compare with*
  - **Implementation:** *Identical implementation to that of [DiT](https://github.com/facebookresearch/DiT)*

- UViT
  - **Location:** ```/models/uvit.py```
  - **Relevant Modules:** None
  - **Run Script:** ```run_diffusion_uvit.py```
  - **Description:** *Actually a DiT with UViT based skip connections. There are subtle differences.*
  - **Implementation:** *DiT with UNet structure by storing shallow "encoder" featuresand concatonating them with deep "decoder" features. After concatonation, they are passed through a linear layer for downsampling back to the original hidden dimension size as per the UViT design. The main differences between UDiT and UViT are namely that conditioning is done with additional sequence tokens with UViT while we use adaptive layer normalization modulation like with DiT. *

## Auxilary Code

### Diffusion Code
- Diffusion Pipeline
  - **Location:** ```/diffusion```
  - **Description** *Diffusion Pipelining Code from OpenAI*
- Miscellaneous Modules
  - **Location:** ```/models/modules```
    - ```conditioned_sequential.py``` : *Implementation of nn.Sequential with Conditioning Information*
    - ```embeddings.py``` : *Implementation of time step and class label embeddings from DiT as well as our custom Modulator layer*
    - ```initialize.py``` : *Weight initializers for various and specific layers*
    - ```positional.py``` " *Positional Embeddings from FAIR*
- Validation Code
  - **Location:** ```/runners```, ```/data```
  - **Description** : Validation code to validate the isolated XSwin backbone




### Validation Code
