import torch
print(torch.cuda.is_available())  # Should return True if CUDA is working
print(torch.cuda.get_device_name(0))  # Shows your GPU name
print(torch.version.cuda)  # Shows the CUDA version used by PyTorch
