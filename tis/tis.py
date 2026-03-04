import torch
import torch.nn.functional as F
from torchvision.models import VisionTransformer as VisionVIT
from timm.models.vision_transformer import VisionTransformer as TimmVIT

from fast_pytorch_kmeans import KMeans

from tqdm import tqdm

import math


class TIS:
    def __init__(self,
                 model,
                 n_masks=1024,  # 论文中设置的默认掩码数量
                 batch_size=128,
                 tokens_ratio=0.5,  # 论文中使用的token保留比例
                 normalise=True,
                 verbose=True,
                 ablation_study=False,
                 ):
        """
        创建TIS类以视觉Transform的显著性映射
        ：param model：要解释的ViT模型
        ：param n_masks：用于生成显著性映射的掩码数（论文中的Nm参数）
        ：param batch_size：用于计算掩码分数的批大小
        ：param tokens_ratio：要保留在屏蔽过程中的标记的比率（论文中的Nk参数）
        ：param normalise：Bool，规范化[0,1]之间的显著性映射
        ：param verbose:Bool，在计算期间打印信息
        ：param ablation_study:Bool，如果为True，则使用消融研究模式，这意味着在
        输入图像而不是编码标记

        """

        # Check that model is a ViT，断言 确定模型是VIT
        assert isinstance(model, VisionVIT) or isinstance(model, TimmVIT), "Transformer architecture not recognised"

        # Save model
        self.model = model

        # 参数设置
        self.batch_size = batch_size
        self.n_masks = n_masks  # 论文中的Nm参数
        self.normalise = normalise
        self.verbose = verbose
        self.ablation_study = ablation_study

        #单值转列表 可以接受多个参数
        if isinstance(tokens_ratio, float):
            tokens_ratio = [tokens_ratio]
        self.tokens_ratio = tokens_ratio  # 论文中的Nk参数，控制保留的token比例

        # 初始化工作变量
        self.encoder_activations = []
        self.encoder_hook_list = []
        self.cur_mask_indices = None

    def __call__(self, x, class_idx=None):
        """
        call方法 让本类变成可调用的函数
        调用main函数以生成显著映射
        ：param x：张量（3，将为其生成贴图的输入图像
        ：param class_idx：可选，要探索的类的索引
        如果未指定，将使用模型预测的类
        ：return：显著映射，形状张量（标记h，标记w）
        """

        # Check that we get only one image
        #输入维度断言
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # 如果需要的话取消缩放以获得四个维度
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        # 梯度禁用上下文，在with这一个代码块内禁用梯度计算
        # 可以节省内存和计算资源，生成显著图不需要反向传播
        # 模型推理，参数更新，评估模式都不需要梯度计算
        with torch.no_grad():

            # 第一次前向传播：获取编码器激活值 和 预测类别（对应论文第3.3节）
            predicted_class, encoder_activations = self.get_encoder_activations(x)
            print("激活值的shape:",encoder_activations.shape)
            print("激活值:",encoder_activations)
            # 确定要解释的类别：如果没有指定，使用模型预测的类别
            if class_idx is None:
                class_idx = predicted_class
                if self.verbose:
                    print("class idx", class_idx)

            # 生成掩码：创建用于遮挡图像的掩码（对应论文第3.3节）
            raw_masks = self.generate_raw_masks(encoder_activations)  # 生成原始掩码
            print("原始mask的shape:",raw_masks.shape)
            mask_list, mask_indices_list = self.generate_binary_masks(raw_masks)  # 生成二进制掩码和索引

            # 生成显著性图：计算每个掩码的重要性分数并合成显著性图（对应论文第3.4节）
            scores = self.generate_scores(x, class_idx, mask_indices_list)  # 计算每个掩码的分数（对应公式4）

            saliency_map = self.generate_saliency(x, scores, mask_list)  # 根据分数和掩码生成显著性图（对应公式5）

            return saliency_map  # 返回最终的显著性图

    def get_encoder_activations(self, x):
        """
        Retrieve the encoder activations for a given image x
        获取给定图像x的编码器激活值（对应论文第3.3节，获取所有层的激活值）

        :param x: image as a tensor
        参数x: 图像张量
        :return: tuple of predicted_class (int), encoder_activations (tensor)
        返回: 元组 (预测类别(int), 编码器激活值(张量))
        """
        # Reset activations and hooks lists
        # 重置激活值和钩子列表
        self.encoder_activations = []
        self.encoder_hook_list = []

        # Define the encoder hook function to retrieve the activations
        # 定义编码器钩子函数来获取激活值
        def encoder_hook_fn(_, __, output):
            # Store activations into the encoder_activations list
            # 将激活值存储到encoder_activations列表中
            self.encoder_activations.append(output.detach())

        # 根据不同的ViT模型类型获取层
        if isinstance(self.model, VisionVIT):
            layers = self.model.encoder.layers  # 获取VisionVIT模型的编码器层
        elif isinstance(self.model, TimmVIT):
            layers = self.model.blocks  # 获取TimmVIT模型的块层
        else:
            print("Model not recognised")  # 模型无法识别
            exit(1)  # 退出程序

        # Attach a forward hook to each transformer block
        # 为每个Transformer快注册前向钩子
        for layer in layers:
            self.encoder_hook_list.append(layer.register_forward_hook(encoder_hook_fn))

        # Forward pass: get the predicted class and activations are retrieved using the hooks
        # 前向传播：获取预测类别，同时通过钩子获取激活值
        predicted_class = torch.argmax(self.model(x))  # 获取模型预测的类别索引

        # Concatenate the list of activations into a single tensor
        # 将激活值列表拼接成单个张量（对应论文中A矩阵的构造）
        self.encoder_activations = torch.cat(self.encoder_activations, dim=-1)

        # Remove hooks
        # 移除所有钩子
        for hook in self.encoder_hook_list:
            hook.remove()  # 移除钩子
        self.encoder_hook_list = []  # 清空钩子列表

        return predicted_class, self.encoder_activations  # 返回预测类别和编码器激活值

    def generate_raw_masks(self, encoder_activations):
        """
        Generate the masks based on the activations
        基于编码器激活值生成原始掩码（对应论文第3.3节，公式1）

        参数encoder_activations: 激活值张量
        返回: 原始掩码列表（张量列表）
        """
        # Squeeze to shape (n_tokens+1, n_activations)
        # 压缩维度，形状变为 (token数量+1, 激活值数量)
        # 移除添加的批次维度（token数量（196图像patch + 1个[CLS] token）*特征维度）
        print("计算原始掩码前激活值维度：",encoder_activations.shape)
        encoder_activations = encoder_activations.squeeze(0)
        print("移除批次维度后的激活值维度：",encoder_activations.shape)

        # remove CLS token and transpose, shape (n_activations, n_tokens)
        # 移除CLS令牌并进行转置，形状变为 (激活值数量, token数量)
        encoder_activations = encoder_activations[1:].T

        # Create clusters with kmeans
        # 使用K均值聚类创建簇（对应论文公式1）
        #n_clusters=self.n_masks：聚类数量，对应论文中的掩码数量Nm。每个聚类中心生成一个掩码
        #mode='euclidean'：使用欧几里得距离作为相似度度量
        #verbose=self.verbose：是否显示聚类过程信息

        # 1. 检查输入形状
        print(f"🎯 聚类输入数据形状: {encoder_activations.shape}")
        print(f"   - 样本数量: {encoder_activations.shape[0]}")
        print(f"   - 特征维度: {encoder_activations.shape[1]}")
        kmeans = KMeans(n_clusters=self.n_masks, mode='euclidean', verbose=self.verbose)
        kmeans.fit(encoder_activations)  # 对激活值进行聚类

        # 4. 检查聚类结果形状
        print(f"📊 聚类结果:")
        print(f"   - 质心形状: {kmeans.centroids.shape}")
        print(f"   - 质心数量: {len(kmeans.centroids)}")
        print(f"   - 每个质心的维度: {kmeans.centroids.shape[1]}")
        # Use kmeans centroids as basis for masks
        # 使用K均值的质心作为掩码的基础（对应论文中的K矩阵）
        raw_masks = kmeans.centroids  # 获取聚类质心

        return raw_masks  # 返回原始掩码


    def generate_binary_masks(self, raw_masks):
        """
        Generate binary masks based on the raw masks
        基于原始掩码生成二进制掩码（对应论文第3.3节，公式2）

        :param raw_masks: list of raw masks
        参数raw_masks: 原始掩码列表
        :return: tuple (mask_list, mask_indices_list)
        返回: 元组 (掩码列表, 掩码索引列表)
        mask_list is a list of masks (list of tensors)
        掩码列表是掩码的列表（张量列表）
        mask_indices_list is a list of indices for each mask (list of tensors)
        掩码索引列表是每个掩码的索引列表（张量列表）
        """

        # Initialise lists for the masks
        # 初始化掩码列表
        mask_indices_list = []  # 掩码索引列表
        mask_list = []  # 二进制掩码列表

        # 遍历不同的token保留比例，依次进行参数计算
        for ratio in self.tokens_ratio:
            # 遍历每个原始掩码
            for raw_mask in raw_masks:
                # Computer the number of tokens to keep based on the ratio
                # 根据比例计算要保留的令牌数量（对应论文中的Nk参数）
                n_tokens = int(ratio * raw_mask.flatten().shape[0])

                # Compute the indexes of the n_tokens with the highest values in the raw mask
                # 计算原始掩码中值最高的n_tokens个令牌的索引（对应论文公式2中的topk操作）
                mask_indices = raw_mask.topk(n_tokens)[1]  # topk返回(值, 索引)

                # Create binary mask
                # 创建二进制掩码（对应论文中的M矩阵）
                bin_mask = torch.zeros_like(raw_mask)  # 创建与原始掩码同形状的零张量
                bin_mask[mask_indices] = 1  # 将选中的索引位置设为1

                # Append current mask to lists
                # 将当前掩码添加到列表中
                mask_indices_list.append(mask_indices)  # 添加掩码索引
                mask_list.append(bin_mask)  # 添加二进制掩码

        return mask_list, mask_indices_list  # 返回掩码列表和索引列表

    def generate_scores(self, x, class_idx, mask_indices_list):
        """
        Generate the masks scores
        生成掩码分数（对应论文第3.4节，公式4）

        :param x: Image to produce the saliency map
        参数x: 用于生成显著性图的图像
        :param class_idx: Class to explore for the saliency map
        参数class_idx: 要探索的类别索引
        :param mask_list: List of masks (list of tensors)
        参数mask_list: 掩码列表（张量列表）
        :param mask_indices_list: List of masks indices (list of tensors)
        参数mask_indices_list: 掩码索引列表（张量列表）
        :return: Score tensor
        返回: 分数张量
        """
        # initialise the list of scores of the masks
        # 初始化掩码分数列表
        scores = []

        # Reset self.cur_mask_indices
        # 重置当前掩码索引
        self.cur_mask_indices = None

        # Define the hook to sample tokens based on the current masks
        # 定义基于当前掩码采样tokens的钩子函数
        # It is designed to receive only one set of tokens as input (batch of size 1) and output a batch with the same
        # size as the number of set of masks indices in the list self.cur_mask_indices
        # 设计为接收单组tokens输入（批次大小为1），输出批次大小等于self.cur_mask_indices中掩码索引集的数量
        def tokens_sampling_hook_fn(_, __, output):
            # Perform sampling only if a mask is set
            # 仅在设置了掩码时执行采样
            if self.cur_mask_indices is not None:
                # Separate CLS from other tokens
                # 分离CLS token和其他token
                cls = output[:, 0].unsqueeze(1)  # 提取CLS token并保持维度
                tokens = output[:, 1:]  # 提取图像块token

                # List of sampled tokens in the batch
                # 批次中采样tokens的列表
                sampled_tokens = []

                # Iterate over the masks of the batch
                # 遍历批次中的掩码
                for indices in self.cur_mask_indices:
                    # Sample the tokens based on current mask indices
                    # 基于当前掩码索引采样tokens（对应论文公式3）
                    cur_tokens = tokens[:, indices]
                    # Add again CLS token
                    # 重新添加CLS令牌
                    cur_tokens = torch.cat([cls, cur_tokens], dim=1)
                    # Add to the list of sampled tokens
                    # 添加到采样tokens列表
                    sampled_tokens.append(cur_tokens)

                # Concatenate the list into a tensor
                # 将列表连接成张量
                sampled_tokens = torch.cat(sampled_tokens)

                return sampled_tokens

        # 如果不是消融研究模式，注册采样钩子
        if not self.ablation_study:
            # Register the sampling hook at the beginning of the encoder, after the positional embedding
            # 在编码器开始处（位置编码之后）注册采样钩子
            if isinstance(self.model, VisionVIT):
                tokens_sampling_hook = self.model.encoder.dropout.register_forward_hook(tokens_sampling_hook_fn)
            elif isinstance(self.model, TimmVIT):
                tokens_sampling_hook = self.model.pos_drop.register_forward_hook(tokens_sampling_hook_fn)
            else:
                print("Model not recognised")
                exit(1)

        # Compute scores by batch
        # 按批次计算分数
        for idx in tqdm(range(math.ceil(len(mask_indices_list) / self.batch_size)), disable=(not self.verbose)):
            # Select the masks attributed to the current batch
            # 选择属于当前批次的掩码
            selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, len(mask_indices_list)))
            self.cur_mask_indices = mask_indices_list[selection_slice]

            if self.ablation_study:
                # Mask the input
                # 消融研究模式：掩码输入图像
                result = self.model(self.mask_input(x)).detach()
            else:
                # Forward pass with tokens sampling performed by the hook
                # 前向传播，通过钩子执行tokens采样
                result = self.model(x).detach()

            # Get the softmax result for the explored class
            # 获取探索类别的softmax结果
            result = torch.softmax(result, dim=1)
            score = result[:, class_idx]  # 提取目标类别的概率分数（对应论文公式4中的w_j,c）

            # Append the scores of the masks in the batch to the list of all scores
            # 将批次中掩码的分数添加到总分数列表
            scores.append(score)

        # Remove sampling hook
        # 移除采样钩子
        if not self.ablation_study:
            tokens_sampling_hook.remove()
        self.cur_mask_indices = None  # 重置当前掩码索引

        # Concatenate all the scores into a tensor
        # 将所有分数连接成张量
        scores = torch.cat(scores)

        return scores  # 返回分数张量

    def generate_saliency(self, x, scores, mask_list):
        """
        生成最终的显著性图
        Generate the final saliency map（对应论文第3.4节，公式5）
        """

        # Stack masks into a tensor
        # 将掩码列表堆叠成张量
        # 形状: (n_tokens, n_masks) - 每个token对应每个掩码的值
        masks = torch.vstack(mask_list).T

        # Sum the masks weighted by their scores to produce a raw saliency
        # 将掩码按分数加权求和，生成原始显著性图（对应论文公式5的分子部分）
        # scored_masks形状: (n_tokens, n_masks) - 每个掩码乘以其分数
        scored_masks = scores * masks
        # raw_saliency形状: (n_tokens,) - 对所有掩码维度求和
        raw_saliency = scored_masks.sum(-1)

        # Compute tokens coverage bias
        # 计算token覆盖偏差（每个token被多少掩码覆盖）（对应论文公式5的分母部分）
        # coverage_bias形状: (n_tokens,) - 每个token在所有掩码中被激活的次数
        coverage_bias = masks.sum(-1)

        # 根据模型类型获取patch大小
        if isinstance(self.model, VisionVIT):
            patch_size = (self.model.patch_size, self.model.patch_size)
        elif isinstance(self.model, TimmVIT):
            patch_size = self.model.patch_embed.patch_size
        else:
            print("Model not recognised")
            exit(1)

        # Compute the saliency map height and width
        # 计算显著性图的高度和宽度
        # 基于原始图像尺寸和patch大小计算
        h = x.shape[-2] // patch_size[0]  # 高度方向patch数量
        w = x.shape[-1] // patch_size[1]  # 宽度方向patch数量

        # Correct the saliency for coverage bias and reshape in 2D
        # 使用覆盖偏差校正显著性图，并重塑为2D（对应论文公式5）
        # saliency = raw_saliency / coverage_bias  # 平均每个掩码的贡献
        saliency = raw_saliency / coverage_bias  # 对应论文公式5：TIS_c = (Σ w_j,c * M_,j) ⊘ (Σ M_,j)
        # 将1D的token序列重塑为2D网格
        saliency = saliency.reshape(h, w)

        # Normalise between [0,1] if self.normalise is True
        # 如果启用标准化，将显著性图归一化到[0,1]范围
        if self.normalise:
            saliency = saliency - saliency.min()  # 平移使最小值为0
            saliency = saliency / saliency.max()  # 缩放使最大值为1

        return saliency  # 返回最终的显著性图