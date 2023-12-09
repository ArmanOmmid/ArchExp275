# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
from models.xswin_diffusion import XNetSwinTransformerDiffusion
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)

    # Seed setup
    torch.manual_seed(args.global_seed)

    # Load model:
    latent_size = int(args.image_size) // 8
    num_classes = args.num_classes

    # Load model:
    patch_size = [1, 1]
    embed_dim = 192
    depths = [3, 3]
    num_heads = [6, 12]
    window_size = [2, 2]

    global_stages = 3
    input_size = [latent_size, latent_size]
    final_downsample = False
    residual_cross_attention = False
    input_channels = 4
    output_channels = 8
    class_dropout = 0.1
    smooth_conv = True

    model = XNetSwinTransformerDiffusion(patch_size, embed_dim, depths,
                            num_heads, window_size, num_classes=num_classes,
                            global_stages=global_stages, input_size=input_size,
                            final_downsample=final_downsample, residual_cross_attention=residual_cross_attention,
                            input_channels=input_channels, output_channels=output_channels,
                            class_dropout_prob=class_dropout, smooth_conv=smooth_conv,
                            ).to(device)

    # Load checkpoint if provided
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

    model.eval()  # Set the model to evaluation mode
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    # os.makedirs(args.sample_dir, exist_ok=True)
    # print(f"Saving .png samples at {args.sample_dir}")
    model_string_name = "xswin"
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Sampling
    total_samples = args.num_fid_samples
    n = args.per_proc_batch_size
    iterations = int(math.ceil(total_samples / n))
    for iteration in tqdm(range(iterations), desc="Sampling images"):
        # Sample inputs:
        z = torch.randn(n, input_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (n,), device=device)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model, z.shape, z, clip_denoised=False, model_kwargs=dict(y=y), progress=False, device=device
        )
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = iteration * n + i
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--xswin", type=str, default=False,
                        help="Whether we are using an XSwin Model")

    args = parser.parse_args()
    # args = parser.parse_args(["--sample-dir", "/content/samples",
    #                           "--image-size", "64",
    #                           "--num-fid-samples", "50000",
    #                           "--num-classes", "101",
    #                           "--ckpt", "/content/drive/MyDrive/XSwinDiffusion_Checkpoints/003-xswin/checkpoints/0107500.pt",
    #                           "--num-sampling-steps", "150",
    #                           "--per-proc-batch-size", "512"])


    main(args)
