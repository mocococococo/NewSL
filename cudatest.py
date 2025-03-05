import torch

# CUDAとcuDNNが有効か確認
print("CUDA available: ", torch.cuda.is_available())
print("cuDNN enabled: ", torch.backends.cudnn.enabled)
