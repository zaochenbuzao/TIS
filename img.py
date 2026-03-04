import numpy as np
import matplotlib.pyplot as plt
import os


def save_saliency_maps(npz_file_path, output_dir='saliency_maps', num_images=10):
    """
    将npz文件中的热力图保存为图像文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data = np.load(npz_file_path)
    saliency_maps = data['arr_0']  # 根据实际情况调整键名

    # 保存前num_images张图
    for i in range(min(num_images, len(saliency_maps))):
        plt.figure(figsize=(8, 6))
        plt.imshow(saliency_maps[i], cmap='hot')
        plt.colorbar()
        plt.title(f'Saliency Map {i + 1}')
        plt.axis('off')

        # 保存图像
        output_path = os.path.join(output_dir, f'saliency_map_{i + 1}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"已保存: {output_path}")


# 使用示例
save_saliency_maps('./npz/vit_b16_tis.npz', 'output_saliency_maps', 10)