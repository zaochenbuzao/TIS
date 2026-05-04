import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from PIL import Image

def load_samples(dataset_root, yaml_path):
    """构建全局样本列表，按子集索引返回前100个（必须与generate.py排序方式一致）"""
    class_to_idx = {}
    samples = []
    for cls_name in sorted(os.listdir(dataset_root)):
        cls_path = os.path.join(dataset_root, cls_name)
        if os.path.isdir(cls_path):
            class_to_idx[cls_name] = len(class_to_idx)
            # 关键：必须与 generate.py 一致，此处使用 sorted 确保确定性
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_path, img_file)
                    samples.append((img_path, class_to_idx[cls_name]))

    with open(yaml_path, 'r') as f:
        subset_indices = yaml.safe_load(f)

    first_100_indices = subset_indices[:100]

    selected_samples = []
    for idx in first_100_indices:
        if idx < 0 or idx >= len(samples):
            raise IndexError(f"Index {idx} out of range (0, {len(samples)-1})")
        selected_samples.append(samples[idx])

    return selected_samples, first_100_indices

def main():
    # 路径配置
    dataset_root = "../inputs/imagenet/val"
    yaml_path = "../inputs/imagenet/indices/sub5000.yaml"
    npz_path = "saliency_maps.npz"
    output_dir = "visual_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载前100个样本
    print("Loading sample list...")
    selected_samples, indices = load_samples(dataset_root, yaml_path)
    print(f"Selected {len(selected_samples)} samples (first 100 from sub5000)")

    # 2. 加载显著性图
    print("Loading saliency maps...")
    data = np.load(npz_path)
    saliency_maps = data['saliency_maps']  # shape (5000, H_s, W_s)
    if saliency_maps.shape[0] < 100:
        raise ValueError(f"Expected at least 100 saliency maps, got {saliency_maps.shape[0]}")
    saliency_maps_100 = saliency_maps[:100]  # 对应前100个索引

    # 3. 生成高清可视化
    print("Generating high-resolution visualizations...")
    cmap = plt.get_cmap('jet')

    for i, (img_path, class_idx) in enumerate(selected_samples):
        # 读取原图
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # 当前热力图
        hm = saliency_maps_100[i]

        # 归一化
        hm_min, hm_max = hm.min(), hm.max()
        hm_norm = (hm - hm_min) / (hm_max - hm_min + 1e-8)

        # 上采样热力图至原图尺寸
        hm_img = Image.fromarray((hm_norm * 255).astype(np.uint8))
        hm_img = hm_img.resize((orig_w, orig_h), Image.BILINEAR)
        hm_resized = np.array(hm_img) / 255.0

        # 彩色热力图
        colored_hm = cmap(hm_resized)[:, :, :3]

        # 原图 numpy
        img_np = np.array(img).astype(np.float32) / 255.0

        # 叠加
        alpha = 0.5
        overlay_np = img_np * (1 - alpha) + colored_hm * alpha
        overlay_np = np.clip(overlay_np, 0, 1)

        # 转为 uint8 拼接
        left = (img_np * 255).astype(np.uint8)
        middle = (colored_hm * 255).astype(np.uint8)
        right = (overlay_np * 255).astype(np.uint8)

        combined = np.hstack([left, middle, right])
        out_path = os.path.join(output_dir, f"sample_{i:04d}.png")
        Image.fromarray(combined).save(out_path)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(selected_samples)}")

    print(f"Done! High-resolution visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()