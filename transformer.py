from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from utils import l2norm, default, exists


# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned
# 2D 正弦位置编码
# 简单的 ViT 论文表明，与学习得到的位置编码相比，这种方法已经足够好

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    """
    生成二维正弦位置编码。

    该函数根据输入的图像块（patches）生成二维正弦位置编码，类似于 Transformer 中的位置编码。
    这种方法使用正弦和余弦函数来编码位置信息，而不需要学习得到的位置编码。

    参数:
        patches (torch.Tensor): 输入的图像块张量，形状为 (N, H, W, D)。
        temperature (float, 可选): 控制位置编码频率的温度参数，默认为 10000。
        dtype (torch.dtype, 可选): 位置编码的数据类型，默认为 torch.float32。

    返回:
        torch.Tensor: 生成的位置编码张量，形状为 (H, W, D)。
    """
    # 获取输入张量的形状、设备和数据类型
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 使用 torch.meshgrid 生成二维网格坐标 (y, x)
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    # 确保特征维度是 4 的倍数，因为正弦和余弦各占一半
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    # 生成 omega 数组，用于控制位置编码的频率
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    # 计算频率
    omega = 1. / (temperature ** omega)

    # 将 y 和 x 坐标与 omega 相乘，生成位置编码的基础
    y = y.flatten()[:, None] * omega[None, :] # y 坐标的位置编码
    x = x.flatten()[:, None] * omega[None, :] # x 坐标的位置编码

    # 生成正弦和余弦位置编码
    # 拼接正弦和余弦编码
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    # 转换数据类型
    pe = pe.type(dtype)

    # 重塑张量形状为 (H, W, D)
    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)


# biasless layernorm
# 无偏置的层归一化

class LayerNorm(nn.Module):
    """
    无偏置的层归一化（Layer Normalization）模块。

    该类实现了层归一化，但移除了偏置（beta）参数，只保留缩放（gamma）参数。
    如果不需要缩放，可以将 `scale` 参数设置为 False。

    参数:
        dim (int): 输入张量的最后一个维度，用于层归一化。
        scale (bool, 可选): 是否使用缩放参数，默认为 True。
    """
    def __init__(self, dim, scale = True):
        super().__init__()
        # 如果需要缩放，则创建一个可学习的 gamma 参数；否则，设置为 None
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        # 注册 gamma 和 beta 缓冲区，persistent=False 表示这些缓冲区不会被保存到模型状态中
        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        """
        前向传播方法，执行层归一化操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 层归一化后的张量。
        """
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)


# feedforward

class GEGLU(nn.Module):
    """
    GEGLU 激活函数模块。

    GEGLU（GeLU Gate Linear Unit）是一种门控线性单元激活函数，它将输入张量分成两部分：
    一部分通过 GeLU 激活函数，另一部分作为门控信号。
    最终输出是 GeLU 激活后的部分与门控信号的乘积。

    参考文献:
        - Hendrycks, D., & Gimpel, K. (2020). Gaussian Error Linear Units (GELUs).
    """
    def forward(self, x):
        """
        前向传播方法，执行 GEGLU 激活函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., dim)。

        返回:
            torch.Tensor: 经过 GEGLU 激活函数处理后的张量，形状为 (..., dim / 2)。
        """
        # 将输入张量沿最后一个维度分成两部分
        x, gate = x.chunk(2, dim = -1)
        # 对 gate 部分应用 GeLU 激活函数，并与 x 部分相乘
        return F.gelu(gate) * x


def FeedForward(dim, mult = 4, dropout = 0.):
    """
    构建前馈神经网络模块。

    该函数构建一个前馈神经网络模块，包含层归一化、线性层、GEGLU 激活函数、Dropout 和输出线性层。
    隐藏层的维度是输入维度的 `mult * 2 / 3` 倍。

    参数:
        dim (int): 输入张量的维度。
        mult (int, 可选): 隐藏层维度的倍数，默认为 4。
        dropout (float, 可选): Dropout 概率，默认为 0。

    返回:
        nn.Sequential: 包含前馈神经网络各层的有序容器。
    """
    # 计算隐藏层的维度
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim), # 层归一化
        nn.Linear(dim, dim_hidden * 2, bias = False), # 线性层，输出维度为隐藏层的两倍
        GEGLU(), # GEGLU 激活函数
        nn.Dropout(dropout), # Dropout 层
        nn.Linear(dim_hidden, dim, bias = False) # 输出线性层，输出维度与输入维度相同
    )


class Attention(nn.Module):
    """
    多头自注意力机制模块。

    该模块实现了多头自注意力机制，用于捕捉输入序列中的长距离依赖关系。
    支持因果掩码（因果自注意力）和相对位置偏置。

    参数:
        dim (int): 输入张量的维度。
        causal (bool, 可选): 是否使用因果掩码，默认为 False。
        dim_head (int, 可选): 每个注意力头的维度，默认为 64。
        heads (int, 可选): 注意力头的数量，默认为 8。
        dropout (float, 可选): Dropout 概率，默认为 0。
        scale (int, 可选): 缩放因子，用于调整相似度分数，默认为 8。
    """
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads  # 注意力头的数量
        self.scale = scale  # 缩放因子
        self.causal = causal  # 是否使用因果掩码
        inner_dim = dim_head * heads  # 内部维度

        # 层归一化
        self.norm = LayerNorm(dim)

        # 注意力 Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # 查询线性变换
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 键和值线性变换
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 查询缩放因子参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        # 键缩放因子参数
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False), # 输出线性变换
            nn.Dropout(dropout) # 输出 Dropout
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        """
        前向传播方法，执行多头自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, N, D)。
            rel_pos_bias (Optional[torch.Tensor]): 相对位置偏置，可选。
            mask (Optional[torch.Tensor]): 掩码，可选。

        返回:
            torch.Tensor: 经过多头自注意力机制处理后的输出张量。
        """
        # 获取批次大小 (B)、序列长度 (N) 和设备
        b, n, _, device = *x.shape, x.device

        # prenorm
        # 前置层归一化
        # 对输入张量进行层归一化
        x = self.norm(x)

        # project for queries, keys, values
        # 投影查询 (q)、键 (k) 和值 (v)
        # 使用线性变换将输入投影到查询、键和值
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention
        # 分割为多头注意力
        # 重塑张量形状以适应多头注意力
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # qk rmsnorm, technique circulating within brain used to stabilize a 22B parameter vision model training
        # qk RMSNorm，稳定训练过程的技术
        # 对查询和键进行 L2 归一化
        q, k = map(l2norm, (q, k))
        # 缩放查询
        q = q * self.q_scale
        # 缩放键
        k = k * self.k_scale

        # similarities
        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            # 如果存在相对位置偏置，则将其添加到相似度分数中
            sim = sim + rel_pos_bias

        if exists(mask):
            # 重塑掩码形状以适应相似度张量
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # 应用掩码，填充负无穷大
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            # 获取相似度张量的最后两个维度
            i, j = sim.shape[-2:]
            # 创建因果掩码（只保留上三角部分）
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            # 应用因果掩码
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重
        # 对相似度分数进行 softmax 归一化
        attn = sim.softmax(dim = -1)
        # 应用注意力 Dropout
        attn = self.attn_dropout(attn)

        # aggregate
        # 使用注意力权重聚合值
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        # 合并多头
        # 重塑张量形状以合并多头
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 应用输出线性变换和 Dropout
        return self.to_out(out)


# Transformer 架构

class Transformer(nn.Module):
    """
    Transformer 模块。

    该模块实现了 Transformer 架构，包括多个 Transformer 层，每个层包含多头自注意力机制和前馈神经网络。
    支持返回所有层的输出。

    参数:
        dim (int): 输入张量的维度。
        depth (int): Transformer 层的数量。
        dim_head (int, 可选): 每个注意力头的维度，默认为 64。
        heads (int, 可选): 注意力头的数量，默认为 8。
        attn_dropout (float, 可选): 自注意力层的 Dropout 概率，默认为 0。
        ff_mult (int, 可选): 前馈神经网络隐藏层维度的倍数，默认为 4。
        ff_dropout (float, 可选): 前馈神经网络的 Dropout 概率，默认为 0。
    """
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        # 初始化 Transformer 层列表
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 为每一层添加多头自注意力和前馈神经网络
            self.layers.append(nn.ModuleList([
                # 多头自注意力机制
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                # 前馈神经网络
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None,
        return_all_layers = False
    ):
        """
        前向传播方法，执行 Transformer 的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, N, D)。
            rel_pos_bias (Optional[torch.Tensor]): 相对位置偏置，可选。
            mask (Optional[torch.Tensor]): 掩码，可选。
            return_all_layers (bool): 是否返回所有层的输出，默认为 False。

        返回:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]: 
            如果 return_all_layers 为 False，则返回最后一层的输出。
            如果 return_all_layers 为 True，则返回最后一层输出和所有层的输出。
        """
        # 初始化层输出列表
        layers = []

        for attn, ff in self.layers:
            # 多头自注意力机制
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x  # 残差连接
            # 前馈神经网络
            x = ff(x) + x  # 残差连接
            # 将当前层的输出添加到列表中
            layers.append(x)

        if not return_all_layers:
            # 如果不需要返回所有层的输出，则返回最后一层的输出
            return x

        # 如果需要返回所有层的输出，则返回最后一层输出和所有层的输出
        return x, torch.stack(layers[:-1]) if len(self.layers)>1 else None
