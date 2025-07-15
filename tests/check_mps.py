import torch

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_macos13_or_newer())
print(torch.backends.mps.is_macos_or_newer(minor=0, major=0))
print(torch.backends.mps.is_built())