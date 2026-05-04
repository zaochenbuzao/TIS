import numpy as np
import torch
from tqdm import tqdm
import os
import random
import yaml
from PIL import Image
from similarity_graph import SimilarityGraph

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_directory_if_not_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # ========== 配置参数 ==========
    model_name = 'vit'           # 'vit' 或 'deit'
    device = 'cuda'               # 'cpu' 或 'cuda'
    token_ratio = 0.5
    masks_layers = 4
    starting_layer = 0
    no_target = False            # True 表示忽略给定标签，使用模型预测的类别
    seed = 42
    output_npz = 'saliency_maps.npz'
    # 数据集根目录（Linux 路径，两个点返回上一级目录）
    dataset_root = "../inputs/imagenet/val"
    # 子集索引文件路径
    yaml_path = "../inputs/imagenet/indices/sub5000.yaml"
    # ==============================

    seed_everything(seed)

    # 1. 初始化模型
    print(f"Loading model: {model_name} on {device}")
    model = SimilarityGraph(model_name, device)

    # 2. 读取子集索引
    with open(yaml_path, 'r') as f:
        subset_indices = yaml.safe_load(f)
    if not isinstance(subset_indices, list):
        raise ValueError("YAML file must contain a list of integer indices.")
    print(f"Loaded {len(subset_indices)} indices from {yaml_path}")

    # 3. 构建完整样本列表，以按顺序映射索引
    class_to_idx = {}
    samples = []
    for cls_name in sorted(os.listdir(dataset_root)):
        cls_path = os.path.join(dataset_root, cls_name)
        if os.path.isdir(cls_path):
            class_to_idx[cls_name] = len(class_to_idx)
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_path, img_file)
                    samples.append((img_path, class_to_idx[cls_name]))

    if len(samples) == 0:
        raise RuntimeError(f"No images found in {dataset_root}")

    print(f"Found {len(samples)} images in {len(class_to_idx)} classes")

    # 4. 根据子集索引抽取样本
    selected_samples = []
    for idx in subset_indices:
        if idx < 0 or idx >= len(samples):
            raise IndexError(f"Index {idx} out of range (0, {len(samples)-1})")
        selected_samples.append(samples[idx])
    print(f"Selected {len(selected_samples)} samples based on provided indices")

    # 5. 循环生成显著性图
    saliency_maps_list = []
    for img_path, class_idx in tqdm(selected_samples, desc="Computing saliency maps"):
        if no_target:
            class_idx = None

        # 使用位置参数调用，与 Usage_example.py 保持一致
        if class_idx is None:
            saliency, _ = model.get_saliency(img_path, token_ratio, masks_layers, starting_layer)
        else:
            saliency, _ = model.get_saliency(img_path, token_ratio, masks_layers, starting_layer, class_idx)

        # 确保 saliency 是二维 (H, W)
        if saliency.ndim == 3 and saliency.shape[0] == 1:
            saliency = saliency[0]
        saliency_maps_list.append(saliency)

    # 6. 堆叠并保存
    saliency_maps = np.stack(saliency_maps_list, axis=0)   # shape: (N, H, W)
    create_directory_if_not_exists(output_npz)
    np.savez(output_npz, saliency_maps=saliency_maps)
    print(f"Saved saliency maps to {output_npz}")

if __name__ == "__main__":
    main()