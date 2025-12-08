import torch
torch.cuda.empty_cache()
import torch
print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
print(f"当前占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"剩余显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
device = torch.device('cuda:0')
x = torch.randn(1000, 1000).to(device)
y = x @ x.T
print(f"计算后显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")