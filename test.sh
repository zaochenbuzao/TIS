#!/bin/bash

#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=TIS部分实验
#SBATCH --output=TIS_%j.log

# 切换到工作目录
cd /home/zaochenbuzao/workdir/TIS


# 显示当前目录确认
echo "当前工作目录: $(pwd)"

# 初始化conda
source ~/.bashrc

# 激活名为TTT的conda环境
conda activate TTT

# 检查GPU状态
echo "=== GPU信息 ==="
nvidia-smi

# 检查环境和目录内容
echo "=== 环境检查 ==="
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "工作目录: $(pwd)"
echo "目录内容:"
ls -la

# 检查Python环境
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 运行主要的生成脚本

echo "=== 运行chefer1生成脚本 ==="
HYDRA_FULL_ERROR=1 python generate.py  model=vit method=chefer1


echo "=== 作业完成 ==="