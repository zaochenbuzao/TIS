# test_vitcx.py
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "comparison_methods/ViTCX/ViT_CX"))

try:
    from ViT_CX import ViT_CX, reshape_function_vit
    print("✓ ViTCX模块导入成功")
except ImportError as e:
    print(f"✗ ViTCX模块导入失败: {e}")