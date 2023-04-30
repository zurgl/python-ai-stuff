from stable_diffusion_videos import StableDiffusionWalkPipeline

fps = 32
img = fps * 10

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
).to("cuda")

video_path = pipeline.walk(
    prompts=[
      "A painting of a black horse walking inside a celtic forest",
      "A painting of a black horse outside outside a celtic forest",
      "A painting of a black horse meeting a princess near at the border of a celtic forest",
    ],
    seeds=[42, 1337, 257],
    num_interpolation_steps=img,
    height=512,
    width=512,
    output_dir='/home/zu/generated',
    name='horse',
    guidance_scale=8.5,
    num_inference_steps=50,
)
