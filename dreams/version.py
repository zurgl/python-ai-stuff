import torch

print(torch.__version__)
assert torch.cuda.is_available(), "You need a GPU to run this script."
device = "cuda"
torch.manual_seed(1337)
