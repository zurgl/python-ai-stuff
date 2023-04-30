from stable_diffusion_videos import RealESRGANModel

model = RealESRGANModel.from_pretrained('nateraw/real-esrgan')
model.upsample_imagefolder('in_img/', 'out_img/')
