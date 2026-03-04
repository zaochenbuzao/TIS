# Transformer Input Sampling (TiS)

## Introduction
This repository contains the source code for Transformer Input Sampling (TiS) method.
The method produces saliency maps for vision transformers using token masking.
The activations of the network are used to produce binary masks for the tokens.
仓库包含的是TIS方法的源代码，TIS使用了token掩码为视觉Transformer生成显著图，网络的激活用于为token生成二进制掩码
## Requirements
A requirements.txt file is provided to install the necessary libraries to use this method.

## Requirements for comparison
This repository also hold scripts to benchmark in comparison to other methods.
仓库也提供了与其他方法做对比的benchmark代码
A script is provided to configure the comparison methods repositories ("comparison_methods/configure_comparison.sh").
The requirements specific to comparison methods can be installed using requirements_comparison.txt.
The ImageNet 2012 validation dataset can be downloaded on the official ImageNet website. After login, go to the downloads page (https://image-net.org/download-images.php) and after clicking on ILSVRC2012, download the 'Development kit (Task 1 & 2)' as well as the 'Validation images (all tasks)'. By default, both files should be placed under a folder "inputs/imagenet/", it can be changed in the hydra dataset config.

## Usage
The method is provided ready to use as a script, a notebook, or can be used in any project as a library.
You need to install the dependencies listed in Section 'Requirements'.

### Demonstration script
It can be used on an arbitrary image with the following command line:
TIS可以使用下面的命令来运行在任意图像上
```python tis_example.py input_file=image.jpg```, by replacing 'image.jpg' with your image.

默认情况下，结果图像会覆盖显示。也可以使用如下指令来进行保存
Instead of displaying the result, you can save it in a file by using and 'output_file' argument as so:

```python tis_example.py input_file=image.jpg output_file=output.png```

If not specified with a class_idx argument, the class used is the maximum output of the model.
如果未使用class_idx参数指定，则使用的类别是模型的输出（最大概率）。
该脚本使用hydra，因此可以在命令行中更改配置文件（在config文件夹中）中的任何参数。
这里是一个批次大小为16的示例，使用DeiT模型并生成德国牧羊犬（235）的解释。
This script uses hydra, so any parameter from the configuration files (in config folder) can be changed in the command line.
Here is an example with a batch size of 16, using a DeiT model and generating an explanation for German shepherd (235).

```python tis_example.py input_file=image.jpg method.init.batch_size=32 model=deit class_idx=235```

Additionally, this script is compatible with the compared methods (see the 'Requirements for comparison' Section), 
for example using RISE:

```python tis_example.py input_file=image.jpg method=rise```

### Notebook
A jupyter notebook is provided as ```TIS_test.ipynb``` and offers the opportunity to play in live with the method.
It requires the Imagenet validation set by default, but can be easily adapted to an arbitrary image.


### Import in any project
The method can be used as a library by importing this repository for your project.
Here is an example displaying a typical usage.

``` 
from torchvision import transforms

# Load a ViT model 
#加载模型
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda()
model.eval()

# Set tranforms, normalise to ImageNet train mean and sd  
#设置Transformer，图片归一参数。训练权重和随机种子
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])

# Get image 
#获取图
from PIL import Image 
image = Image.open("dog.png").convert('RGB') 
input_tensor = transform(image).cuda()

# Initialize the saliency class (adapt the batch_size depending on the available memory)
#初始化显著性类，根据可用内存调整batch_size batch_size是同时训练的批次大小-
from tis import TIS
saliency_method = TIS(model, batch_size=512)
##class_idx可以省略，在这种情况下，将使用最大预测类
# class_idx can be omited, in this case the maximum predicted class will be used
saliency_map = saliency_method(input_tensor, 
                   #class_idx=class_idx
                  ).cpu()
``` 
