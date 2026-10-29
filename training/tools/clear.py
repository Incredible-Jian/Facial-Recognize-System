"""查看并清理 PyTorch CUDA 显存占用（仅在有 GPU 时有意义）。"""

import torch


def main():
    if not torch.cuda.is_available():
        print("当前环境没有可用的 CUDA 设备。")
        return

    torch.cuda.empty_cache()
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"当前占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"保留占用: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000, device=device)
    y = x @ x.T
    print(f"计算后显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    del x, y
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
