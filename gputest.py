import torch


def check_gpu_basic():
    """基础GPU检测"""
    print("=== PyTorch GPU 信息检测 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"可用 GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n--- GPU {i} ---")
            print(f"设备名称: {torch.cuda.get_device_name(i)}")
            print(f"设备能力: {torch.cuda.get_device_capability(i)}")
            print(f"总显存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"已用显存: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"缓存显存: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
    else:
        print("未检测到可用的 GPU 设备")


if __name__ == "__main__":
    check_gpu_basic()