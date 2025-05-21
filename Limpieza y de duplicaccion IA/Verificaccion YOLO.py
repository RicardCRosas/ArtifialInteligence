import torch
import torchvision

print("✅ CUDA:", torch.cuda.is_available())
print("🚀 PyTorch:", torch.__version__)
print("🧩 TorchVision:", torchvision.__version__)
print("🖥️ Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
