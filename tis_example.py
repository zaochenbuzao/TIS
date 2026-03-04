import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
from torch.nn.functional import interpolate
import numpy as np
import random
import os

from datasets.ilsvrc2012 import classes

from PIL import Image

from matplotlib import pyplot as plt


# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def overlay(image, saliency, alpha=0.7, output_file=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    image = image.permute(1, 2, 0)
    saliency = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[:2], mode='bilinear')
    saliency = saliency.squeeze()
    ax[0].imshow(image)
    ax[1].imshow(image)
    ax[1].imshow(saliency, alpha=alpha, cmap='jet')
    if output_file:
        # If 'output_file' path is absolute
        if os.path.isabs(output_file):
            print(f"Saving the output image under {output_file}")
        # Else print the current working directory path + 'output_file'
        else: 
            print(f"Saving the output image under {os.getcwd()}/{output_file}")
        # Save the explanation
        plt.savefig(output_file)
    else:
        plt.show()


@hydra.main(version_base="1.3", config_path="config", config_name="example")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)
   #使用GPU设备
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # 加载模型
    print("Loading model:", cfg.model.name, end="\n\n")
    model = instantiate(cfg.model.init).to(device)
    model.eval()

    # 获取方法
    print("Initializing saliency method:", cfg.method.name, end="\n\n")
    method = instantiate(cfg.method.init, model)

    # Get transformations
    print("Setting transformations", end="\n\n")
    transform = instantiate(cfg.transform)

    # 载入图片
    print("Opening image:", cfg.input_file, end="\n\n")
    #使用PIL库打开输入的图像文件，并将图像转化为三通道
    image_raw = Image.open(cfg.input_file).convert('RGB')
    #transform处理图像 并将其移入设备，在第0维添加一个批次维度（图像一般需要分批次处理）
    image = transform(image_raw).to(device).unsqueeze(0)

    if not cfg.class_idx:
        class_idx = torch.argmax(model(image), dim=-1)[-1]
    else:
        class_idx = cfg.class_idx

    # Computing saliency map,.detach()从计算图中分离，避免梯度追踪.cpu()：将张量从GPU移动到CPU
    print("Computing saliency map using", cfg.method.name, "for class", classes[class_idx])
    saliency_map = method(image, class_idx=class_idx).detach().cpu()

    #把像素值缩放到0-1之间
    image = image - image.min()
    image = image/image.max()

    overlay(image.squeeze(0).cpu(), saliency_map, output_file=cfg.output_file)


if __name__ == "__main__":
    main()
