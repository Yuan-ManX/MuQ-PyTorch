import math
from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
import torch.distributed as dist

from utils import exists, l2norm, log, print_once
from distributed import AllGather
from extend_distributed import all_gather
from transformer import LayerNorm


def matrix_diag(t):
    """
    从输入张量中提取对角线元素。

    该函数从输入张量中提取所有二维子矩阵的对角线元素。
    适用于批量处理的张量。

    参数:
        t (torch.Tensor): 输入张量，形状为 (..., i, j)。

    返回:
        torch.Tensor: 对角线元素张量，形状为 (..., min(i, j))。
    """
    device = t.device  # 获取设备
    i, j = t.shape[-2:]  # 获取最后两个维度的大小
    num_diag_el = min(i, j)  # 计算对角线元素的数量
    i_range = torch.arange(i, device=device)  # 创建行索引张量
    j_range = torch.arange(j, device=device)  # 创建列索引张量

    # 创建对角线掩码，形状为 (i, j)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')

    # 使用掩码提取对角线元素，结果形状为 (..., min(i, j))
    diag_el = t.masked_select(diag_mask)

    # 重塑张量形状为 (..., min(i, j))
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)


# contrastive losses
# 对比学习损失

class SoftmaxContrastiveLearning(nn.Module):
    """
    Softmax 对比学习损失模块。

    该模块实现了 Softmax 对比学习损失，用于对比学习任务。
    支持解耦对比学习（Decoupled Contrastive Learning）。

    参数:
        layers (int, 可选): 对比学习的层数，默认为 1。
        decoupled_contrastive_learning (bool, 可选): 是否使用解耦对比学习，默认为 False。
        init_temp (float, 可选): 初始温度参数，默认为 10。
    """
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        init_temp = 10
    ):
        super().__init__()
        # 初始化温度参数
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        # 是否使用解耦对比学习
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # 初始化 AllGather 模块
        self.all_gather = AllGather(dim = 2)

    @property
    def device(self):
        """
        获取设备。

        返回:
            torch.device: 当前设备。
        """
        return next(self.parameters()).device
    
    def forward(self, audio_latents, text_latents):
        """
        前向传播方法，计算对比学习损失。

        参数:
            audio_latents (torch.Tensor): 音频的潜在表示，形状为 (L, B, D)。
            text_latents (torch.Tensor): 文本的潜在表示，形状为 (L, B, D)。

        返回:
            torch.Tensor: 对比学习损失。
        """
        if audio_latents.ndim == 2:
            # 如果音频潜在表示是二维的，则增加一个维度
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            # 如果文本潜在表示是二维的，则增加一个维度
            text_latents = rearrange(text_latents, '... -> 1 ...')

        # 获取批次大小
        batch = audio_latents.shape[1]

        if self.all_gather.is_distributed:
            # 堆叠音频和文本潜在表示
            latents = torch.stack((audio_latents, text_latents))
            # 使用 AllGather 收集潜在表示
            latents, _ = self.all_gather(latents)
            # 解包收集后的潜在表示
            audio_latents, text_latents = latents

        # 计算相似度矩阵
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)
        # 缩放相似度分数
        sims = sims * self.temperatures.exp()
        # 计算指数相似度矩阵 [Rank, N, N]
        cosine_sims_exp = sims.exp() # Similarity matrix  [Rank, N, N]
        # 提取对角线元素，即正样本的相似度
        numerator = matrix_diag(cosine_sims_exp) # Take diagonal elements, that is, for t [l, i, j], take all elements of i==j to obtain a array of l * min (i, j)

        if self.decoupled_contrastive_learning:
            # 创建单位矩阵掩码
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            # 将对角线元素设置为 0
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.) # Set the diagonal to 0

        # 对每个样本的相似度求和（负样本）
        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        # 计算对比学习损失
        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        # 对批次求平均
        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()


class RankSoftmaxContrastiveLearning(nn.Module):
    """
    基于排名的 Softmax 对比学习损失模块。

    该模块实现了基于排名的 Softmax 对比学习损失，用于对比学习任务。
    支持解耦对比学习（Decoupled Contrastive Learning）。

    参数:
        layers (int, 可选): 对比学习的层数，默认为 1。
        decoupled_contrastive_learning (bool, 可选): 是否使用解耦对比学习，默认为 False。
        init_temp (float, 可选): 初始温度参数，默认为 10。
    """
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        init_temp = 10,
    ):
        super().__init__()
        # 初始化温度参数，使用对数尺度
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        # 是否使用解耦对比学习
        self.decoupled_contrastive_learning = decoupled_contrastive_learning
    
    @property
    def device(self):
        """
        获取设备。

        返回:
            torch.device: 当前设备。
        """
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        """
        前向传播方法，计算基于排名的 Softmax 对比学习损失。

        参数:
            audio_latents (torch.Tensor): 音频的潜在表示，形状为 (L, B, D)。
            text_latents (torch.Tensor): 文本的潜在表示，形状为 (L, B, D)。

        返回:
            torch.Tensor: 对比学习损失。
        """
        if audio_latents.ndim == 2:
            # 如果音频潜在表示是二维的，则增加一个维度
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            # 如果文本潜在表示是二维的，则增加一个维度
            text_latents = rearrange(text_latents, '... -> 1 ...')

        # 使用 AllGather 收集音频潜在表示
        audio_latents = all_gather(audio_latents, None)
        # 使用 AllGather 收集文本潜在表示
        text_latents = all_gather(text_latents, None)

        # 打印潜在表示的形状
        print_once("audio_latents:"+str(audio_latents.shape) + "text_latents:" + str(text_latents.shape))
        
        # 获取批次大小
        batch = audio_latents.shape[1]
        # 获取分布式训练的进程数（排名）
        rank = audio_latents.shape[0]

        # 重塑音频潜在表示形状为 (L * B, D)
        audio_latents = rearrange(audio_latents, 'l i d -> (l i) d')
        # 重塑文本潜在表示形状为 (L * B, D)
        text_latents = rearrange(text_latents, 'l j d -> (l j) d')

        # 计算音频和文本潜在表示之间的相似度
        sims = einsum('i d, j d -> i j', audio_latents, text_latents)
        # 缩放相似度分数
        sims = sims * self.temperatures.exp()
        # 重塑相似度矩阵形状为 (L * B, L * B)
        sims = rearrange(sims, '1 i j -> i j')
        # 计算指数相似度矩阵 [Rank, N, N]
        cosine_sims_exp = sims.exp() # Similarity matrix  [Rank, N, N]
        # 提取对角线元素，即正样本的相似度
        numerator = matrix_diag(cosine_sims_exp) # Take diagonal elements, that is, for t [l, i, j], take all elements of i==j to obtain a array of l * min (i, j)

        if self.decoupled_contrastive_learning:
            # 创建单位矩阵掩码
            eye = torch.eye(batch*rank, device = self.device, dtype = torch.bool)
            # 将对角线元素设置为 0
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.) # Set the diagonal to 0

        # 对每个样本的相似度求和（负样本）
        denominator_i = reduce(cosine_sims_exp, 'i j -> i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'i j -> j', 'sum')

        # 计算对比学习损失
        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))
        # 对批次求平均
        contrastive_loss = reduce(contrastive_loss, '1 n -> 1', 'mean')
        return contrastive_loss


class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """
    """
    基于 Sigmoid 的对比学习损失模块。

    该模块实现了基于 Sigmoid 的对比学习损失，用于语言-图像预训练（SigLIP）。
    与标准的 Softmax 对比学习不同，Sigmoid 损失仅在图像-文本对上操作，不需要全局视角的成对相似度进行归一化。
    Sigmoid 损失允许进一步扩大批量大小，同时在较小的批量大小下也能表现良好。

    参数:
        layers (int, 可选): 对比学习的层数，默认为 1。
        init_temp (float, 可选): 初始温度参数，默认为 10。
        init_bias (float, 可选): 初始偏置参数，默认为 -10。
    """

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        # 初始化温度参数，使用对数尺度
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        # 初始化偏置参数
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

        # 初始化 AllGather 模块，收集张量并聚合梯度
        self.all_gather = AllGather(dim = 1, all_reduce_grads = True)

    @property
    def device(self):
        """
        获取设备。

        返回:
            torch.device: 当前设备。
        """
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        """
        前向传播方法，计算基于 Sigmoid 的对比学习损失。

        参数:
            audio_latents (torch.Tensor): 音频的潜在表示，形状为 (Rank, Batch, D)。
            text_latents (torch.Tensor): 文本的潜在表示，形状为 (Rank, Batch, D)。

        返回:
            torch.Tensor: 基于 Sigmoid 的对比学习损失。
        """
        device = self.device

        if audio_latents.ndim == 2:
            # 如果音频潜在表示是二维的，则增加一个维度 -> [Rank, Batch, Latent]
            audio_latents = rearrange(audio_latents, '... -> 1 ...') # To [Rank, Batch, Latent]

        if text_latents.ndim == 2:
            # 如果文本潜在表示是二维的，则增加一个维度
            text_latents = rearrange(text_latents, '... -> 1 ...')

        # 使用 AllGather 收集文本潜在表示，并获取每个进程的大小
        text_latents, rank_sizes = self.all_gather(text_latents)

        # 获取批次大小
        n = text_latents.shape[1]

        # 计算音频和文本潜在表示之间的点积相似度
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents) # Calculate dot product similarity between pairs
        # 缩放相似度分数并添加偏置
        sims = sims * self.temperatures.exp() + self.bias
        # 创建标签矩阵，对角线元素为 1，其余为 0
        labels = torch.eye(n, device = device)

        if exists(rank_sizes):
            # 根据每个进程的大小分割标签矩阵
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim = 0) 
            # 获取当前进程的标签矩阵
            labels = labels_by_ranks[dist.get_rank()] # labels to the n elements of the current rank

        # 将标签矩阵转换为 -1 和 1，形状为 (1, n, n)
        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        # 计算 Sigmoid 损失
        # labels * sims 为正样本和负样本的标签与相似度的乘积
        # F.logsigmoid 为对数 Sigmoid 函数
        # -F.logsigmoid(labels * sims) 为损失函数
        # sum() 对所有样本求和
        # / n 对损失进行平均
        return -F.logsigmoid(labels * sims).sum() / n


# hierarchical cl loss

def interspersed_indices(layers, total_layers):
    """
    生成在总层数中均匀分布的层索引。

    该函数生成在总层数中均匀分布的层索引，用于分层对比学习。
    例如，如果总层数为 12，层数为 3，则返回 [0, 4, 8]。

    参数:
        layers (int): 需要生成的层数。
        total_layers (int): 总层数。

    返回:
        torch.Tensor: 生成的层索引张量。
    """
    assert total_layers >= layers
    # 计算步长
    step = total_layers / layers
    # 生成层索引，并向下取整
    return (torch.arange(0, layers) * step).floor().long()


class MultiLayerContrastiveLoss(nn.Module):
    """
    分层对比学习损失模块。

    该模块实现了分层对比学习损失，用于在多个层级上计算对比学习损失。
    支持解耦对比学习（Sigmoid 和 Softmax 对比学习）。

    参数:
        audio_dim (int): 音频嵌入向量的维度。
        text_dim (int): 文本嵌入向量的维度。
        dim_latent (int): 潜在空间的维度。
        layers (int): 对比学习的层数。
        decoupled_contrastive_learning (bool, 可选): 是否使用解耦对比学习，默认为 False。
        sigmoid_contrastive_loss (bool, 可选): 是否使用 Sigmoid 对比损失，默认为 False。
    """
    def __init__(
        self,
        *,
        audio_dim,
        text_dim,
        dim_latent,
        layers,
        decoupled_contrastive_learning = False,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        # 保存层数
        self.layers = layers

        # 初始化音频层归一化，不使用缩放
        self.audio_norm = LayerNorm(audio_dim, scale = False)
        # 初始化音频 gamma 参数
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        # 初始化音频到潜在空间的权重
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        # 初始化音频到潜在空间的偏置
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        # 初始化文本层归一化，不使用缩放
        self.text_norm = LayerNorm(text_dim, scale = False)
        # 初始化文本 gamma 参数
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        # 初始化文本到潜在空间的权重
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        # 初始化文本到潜在空间的偏置
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        # 根据是否使用 Sigmoid 对比损失，选择合适的对比学习损失类
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        # 初始化对比学习损失模块
        self.contrast = klass(layers = layers)

    def forward(self, *, audio_layers, text_layers):
        """
        前向传播方法，计算分层对比学习损失。

        参数:
            audio_layers (torch.Tensor): 音频的各层嵌入向量，形状为 (L, B, N, D)。
            text_layers (torch.Tensor): 文本的各层嵌入向量，形状为 (L, B, N, D)。

        返回:
            torch.Tensor: 分层对比学习损失。
        """
        # 获取设备 (device) 和批次大小 (batch)
        device, batch = audio_layers.device, audio_layers.shape[1]

        # 对音频各层嵌入向量进行平均池化，得到 (L, B, D)
        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        # 对音频嵌入向量进行层归一化，并乘以 gamma 参数
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        # 将音频嵌入向量投影到潜在空间，并添加偏置
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        # 对音频潜在表示进行 L2 归一化
        audio_latents = l2norm(audio_latents)

        # 提取文本的分类 token（假设第一个 token 为分类 token）
        text_cls_tokens = text_layers[:, :, 0]
        # 对文本分类 token 进行层归一化，并乘以 gamma 参数
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        # 将文本嵌入向量投影到潜在空间，并添加偏置
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        # 对文本潜在表示进行 L2 归一化
        text_latents = l2norm(text_latents)

        # 计算对比学习损失
        return self.contrast(audio_latents, text_latents)
