import os
import fire
import sys

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import numpy as np
import torch

prompt = "a very realistic chaotic sea storm on south pole"
num_inference_steps = 50
num_steps = 100
max_frames = 1000
guidance_scale = 7.5
seed = 1337
width = 800
height = 600
quality = 100
# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "CompVis/stable-diffusion-v1-4"

outdir = os.path.join(os.getcwd(), "output")
os.makedirs(outdir, exist_ok=True)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def interpolate(t, v0, v1, DOT_THRESHOLD=0.9995):
    inputs_are_torch = False
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def save_img(image, quality, frame_index):
    name = os.path.join(outdir, "frame%03d.jpg" % frame_index)
    image.save(name, format="JPEG", quality=quality)


def model(device, pipe, init):
    with torch.autocast(device):
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            latents=init,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]

    return image


def run():
    assert torch.cuda.is_available(), "You need a GPU to run this script."
    device = "cuda"
    torch.manual_seed(seed)

    lms = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        #torch_dtype=torch.float16,
        #revision="fp16",
        scheduler=lms,
    ).to(device)
    pipe.requires_safety_checker = False
    pipe.safety_checker = None

    init1 = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8), device=device
    )

    frame_index = 0
    while frame_index < max_frames:
        init2 = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8), device=device
        )
        for i, t in enumerate(np.linspace(0, 1, num_steps)):
            init = interpolate(float(t), init1, init2)

            eprint("dreaming... ", frame_index)
            image = model(device, pipe, init)
            save_img(image, quality, frame_index)
            frame_index += 1

        init1 = init2

if __name__ == "__main__":
    fire.Fire(run)
