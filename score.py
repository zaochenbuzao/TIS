import pandas as pd
import numpy as np
import os
import argparse


def calculate_insertion_score(score_sequence):
    """
    计算单个图像的Insertion分数（AUC）

    参数:
    score_sequence: 插入过程中的置信度序列

    返回:
    auc_score: Insertion AUC分数
    """
    # 移除NaN值
    scores = score_sequence[~np.isnan(score_sequence)]

    if len(scores) == 0:
        return 0.0

    # 归一化步数到 [0, 1] 区间
    x = np.linspace(0, 1, len(scores))

    # 计算AUC（梯形法则）
    auc_score = np.trapz(scores, x)

    return auc_score


def process_single_insertion_csv(csv_file_path, output_txt_path):
    """
    处理单个Insertion CSV文件，计算分数并保存结果

    参数:
    csv_file_path: 输入CSV文件路径
    output_txt_path: 输出TXT文件路径
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path, header=None)
        print(f"成功读取文件: {csv_file_path}")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    all_scores = []

    # 处理每一行（每个图像）
    print("正在计算每个图像的Insertion分数...")
    for idx, row in df.iterrows():
        scores = row.values

        # 计算该图像的Insertion分数
        insertion_score = calculate_insertion_score(scores)
        all_scores.append(insertion_score)

    # 转换为numpy数组
    all_scores = np.array(all_scores)

    # 计算统计量
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    median_score = np.median(all_scores)

    # 打印结果
    print("\n=== 计算结果 ===")
    print(f"处理的图像数量: {len(all_scores)}")
    print(f"平均Insertion分数: {mean_score:.6f}")
    print(f"标准差: {std_score:.6f}")
    print(f"中位数: {median_score:.6f}")
    print(f"分数范围: {all_scores.min():.6f} - {all_scores.max():.6f}")

    # 保存详细分数到CSV
    output_csv_path = output_txt_path.replace('.txt', '_scores.csv')
    scores_df = pd.DataFrame({'image_score': all_scores})
    scores_df.to_csv(output_csv_path, index=False)
    print(f"详细分数已保存到: {output_csv_path}")

    # 保存汇总结果到TXT文件
    try:
        with open(output_txt_path, 'w') as f:
            f.write(f"CSV文件名: {os.path.basename(csv_file_path)}\n")
            f.write(f"图像数量: {len(all_scores)}\n")
            f.write(f"平均Insertion分数: {mean_score:.6f}\n")
            f.write(f"标准差: {std_score:.6f}\n")
            f.write(f"中位数: {median_score:.6f}\n")
            f.write(f"最小值: {all_scores.min():.6f}\n")
            f.write(f"最大值: {all_scores.max():.6f}\n")

        print(f"汇总结果已保存到: {output_txt_path}")

        # 在文件末尾添加一行，只包含文件名和平均分数（便于后续批量处理）
        with open(output_txt_path, 'a') as f:
            f.write(f"\n{os.path.basename(csv_file_path)},{mean_score:.6f}")

    except Exception as e:
        print(f"保存结果文件失败: {e}")
        return None

    return {
        'mean_score': mean_score,
        'std_score': std_score,
        'median_score': median_score,
        'min_score': all_scores.min(),
        'max_score': all_scores.max(),
        'num_images': len(all_scores)
    }


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='计算Insertion CSV文件的分数')
    parser.add_argument('csv_file', type=str, help='输入的CSV文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出的TXT文件路径（可选）', default=None)

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"错误: 文件 {args.csv_file} 不存在")
        return

    # 设置输出文件路径
    if args.output is None:
        # 默认输出到同目录下，文件名加_insertion_score
        base_name = os.path.splitext(args.csv_file)[0]
        output_txt_path = f"{base_name}_insertion_score.txt"
    else:
        output_txt_path = args.output

    # 确保输出目录存在
    output_dir = os.path.dirname(output_txt_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理CSV文件
    print(f"开始处理: {args.csv_file}")
    result = process_single_insertion_csv(args.csv_file, output_txt_path)

    if result is not None:
        print(f"\n处理完成！")
        print(f"最终Insertion分数: {result['mean_score']:.6f}")
    else:
        print("处理失败！")


# 如果不使用命令行，可以直接修改下面的文件路径运行
if __name__ == "__main__":
    # 如果直接运行（非命令行），在这里指定文件路径
    # csv_file_path = "your_insertion_results.csv"  # 修改为你的CSV文件路径
    # output_txt_path = "insertion_score_result.txt"  # 修改为想要的输出文件路径

    # 然后注释掉下面这行，取消注释下面两行：
    # process_single_insertion_csv(csv_file_path, output_txt_path)

    # 使用命令行参数
    main()