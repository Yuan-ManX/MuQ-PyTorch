from typing import List, Optional, Union
from collections import OrderedDict
from functools import partial
from torch import nn, einsum

from audio import AudioSpectrogramTransformer, AudioSpectrogramTransformerPretrained
from text import TextTransformer, TextTransformerPretrained
from contrastive import RankSoftmaxContrastiveLearning, SoftmaxContrastiveLearning, SigmoidContrastiveLearning, MultiLayerContrastiveLoss, interspersed_indices
from utils import exists, default, l2norm


class MuLanModel(nn.Module):
    """
    MuLan 模型类。

    该类实现了 MuLan 模型，包括音频和文本 Transformer 的集成、潜在空间的投影以及对比学习损失的计算。
    """
    def __init__(
        self,
        audio_transformer: Union[AudioSpectrogramTransformer, AudioSpectrogramTransformerPretrained],
        text_transformer: Union[TextTransformer, TextTransformerPretrained],
        dim_latent = 128,                       # they use 128
        decoupled_contrastive_learning = True,  # think this was used, make it optional
        hierarchical_contrastive_loss = False,
        hierarchical_contrastive_loss_layers = None,
        sigmoid_contrastive_loss = False,
        rank_contrast = False,    # apply contrast on rank dimension
        proj_to_latent = True,
        norm_type = 'l2norm',
        **kwargs,
    ):
        """
        初始化 MuLan 模型。

        参数:
            audio_transformer (Union[AudioSpectrogramTransformer, AudioSpectrogramTransformerPretrained]): 音频 Transformer 模型。
            text_transformer (Union[TextTransformer, TextTransformerPretrained]): 文本 Transformer 模型。
            dim_latent (int, 可选): 潜在空间的维度，默认为 128。
            decoupled_contrastive_learning (bool, 可选): 是否使用解耦对比学习，默认为 True。
            hierarchical_contrastive_loss (bool, 可选): 是否使用分层对比损失，默认为 False。
            hierarchical_contrastive_loss_layers (Optional[List], 可选): 分层对比损失中使用的层列表，默认为 None。
            sigmoid_contrastive_loss (bool, 可选): 是否使用 sigmoid 对比损失，默认为 False。
            rank_contrast (bool, 可选): 是否在排名维度上应用对比，默认为 False。
            proj_to_latent (bool, 可选): 是否将嵌入向量投影到潜在空间，默认为 True。
            norm_type (str, 可选): 归一化类型，默认为 'l2norm'。
            **kwargs: 其他关键字参数。
        """
        super().__init__()
        # 保存潜在空间的维度
        self.dim_latent = dim_latent

        # audio and text transformer
        # 初始化音频和文本 Transformer
        self.audio = audio_transformer # 音频 Transformer
        self.text = text_transformer # 文本 Transformer

        # two linear layers to project embeddings to latent space
        # 使用两个线性层将嵌入向量投影到潜在空间
        if proj_to_latent:
            # 文本到潜在空间的线性投影
            self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
            # 音频到潜在空间的线性投影
            self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        # 是否使用 sigmoid 对比损失
        self.sigmoid_contrastive_loss = sigmoid_contrastive_loss
        # 是否使用解耦对比学习
        self.decoupled_contrastive_learning = decoupled_contrastive_learning
        # 是否在排名维度上应用对比
        self.rank_contrast = rank_contrast
        # 归一化类型
        self.norm_type = norm_type

        # use decoupled contrastive learning or not, where self.contrast is loss module for contrastive learning
        # 根据参数选择对比学习损失的类型
        if sigmoid_contrastive_loss:
            # 使用 Sigmoid 对比学习损失
            klass = SigmoidContrastiveLearning
        else: 
            if rank_contrast:
                # 使用排名 Softmax 对比学习损失
                klass = partial(RankSoftmaxContrastiveLearning,  decoupled_contrastive_learning = decoupled_contrastive_learning) 
            else:
                # 使用 Softmax 对比学习损失
                klass = partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)

        # 初始化对比学习损失模块
        self.contrast = klass() 

        # 初始化多层对比学习损失模块
        self.multi_layer_contrastive_learning = None

        if hierarchical_contrastive_loss:
            # 如果使用分层对比损失，则计算需要使用的层数
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1) 
            assert num_layers > 0 # 确保层数大于 0

            # 注册音频和文本层的索引缓冲区
            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth)) 
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            # 初始化多层对比学习损失模块
            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim = self.audio.dim,
                text_dim = self.text.dim,
                dim_latent = dim_latent,
                layers = num_layers,
                decoupled_contrastive_learning = decoupled_contrastive_learning,
                sigmoid_contrastive_loss = sigmoid_contrastive_loss
            )

    def get_audio_latents(
        self,
        wavs,
        return_all_layers = False,
    ):
        """
        从音频波形中提取音频嵌入向量和潜在表示。

        参数:
            wavs (torch.Tensor): 输入的音频波形张量。
            return_all_layers (bool, 可选): 是否返回所有层的嵌入向量，默认为 False。

        返回:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: 
            如果 return_all_layers 为 False，则返回音频的潜在表示。
            如果 return_all_layers 为 True，则返回音频的潜在表示和所有层的嵌入向量。
        """
        # 获取音频嵌入向量和所有层的嵌入向量
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers = True)
        # 将音频嵌入向量投影到潜在空间
        audio_latents = self.audio_to_latents(audio_embeds)
        # 对潜在表示进行归一化 -> [Batch, Feat=128]
        out = self._norm_latents(audio_latents) #->[Batch, Feat=128]

        if not return_all_layers:
            # 如果不需要返回所有层的嵌入向量，则返回归一化的潜在表示
            return out
        
        # 如果需要返回所有层的嵌入向量，则返回归一化的潜在表示和所有层的嵌入向量
        return out, audio_layers #[nLayer=5, Batch=2, 15, 512]

    def get_text_latents(
        self,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_all_layers = False
    ):
        """
        从文本输入中提取文本嵌入向量和潜在表示。

        参数:
            texts (Optional[List[str]]): 文本列表，可选。
            raw_texts (Optional[List[str]]): 原始文本列表，可选。
            return_all_layers (bool): 是否返回所有层的嵌入向量，默认为 False。

        返回:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: 
            如果 return_all_layers 为 False，则返回文本的潜在表示。
            如果 return_all_layers 为 True，则返回文本的潜在表示和所有层的嵌入向量。
        """
        # 使用文本 Transformer 提取文本嵌入向量和所有层的嵌入向量
        text_embeds, text_layers = self.text(texts, raw_texts = raw_texts, return_all_layers = True)
        # 将文本嵌入向量投影到潜在空间
        text_latents = self.text_to_latents(text_embeds)
        # 对潜在表示进行归一化
        out = self._norm_latents(text_latents)

        if not return_all_layers:
            # 如果不需要返回所有层的嵌入向量，则返回归一化的潜在表示
            return out

        # 如果需要返回所有层的嵌入向量，则返回归一化的潜在表示和所有层的嵌入向量
        return out, text_layers
    
    def _norm_latents(self, latents):
        """
        对潜在表示进行归一化。

        参数:
            latents (torch.Tensor): 输入的潜在表示。

        返回:
            torch.Tensor: 归一化后的潜在表示。
        """
        if self.norm_type == 'l2norm':
            # 如果归一化类型是 'l2norm'，则使用 L2 归一化
            return l2norm(latents)
        else:
            # 否则，使用默认的归一化方法
            return self.norm(latents)

    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_latents = False,
        return_similarities = False,
        return_pairwise_similarities = False,
    ):
        """
        前向传播方法，计算音频和文本之间的对比学习损失。

        参数:
            wavs (torch.Tensor): 音频波形张量。
            texts (Optional[List[str]]): 文本列表，可选。
            raw_texts (Optional[List[str]]): 原始文本列表，可选。
            return_latents (bool): 是否返回潜在表示，默认为 False。
            return_similarities (bool): 是否返回相似度，默认为 False。
            return_pairwise_similarities (bool): 是否返回成对相似度，默认为 False。

        返回:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
            根据参数返回不同的结果。
            如果 return_latents 为 True，则返回音频和文本的潜在表示。
            如果 return_similarities 为 True，则返回相似度。
            如果 return_pairwise_similarities 为 True，则返回成对相似度。
            否则，返回对比学习损失。
        """
        # 获取批次大小和设备
        batch, device = wavs.shape[0], wavs.device
        
        # both latents are of [Batch, Feat=128]
        # 提取音频和文本的潜在表示，均为 [Batch, Feat=128]
        # 提取音频潜在表示和所有层的嵌入向量
        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers = True)
        # 提取文本潜在表示和所有层的嵌入向量
        text_latents, text_layers = self.get_text_latents(texts, raw_texts = raw_texts, return_all_layers = True)

        # 在推理时使用
        if return_latents: 
            # 返回音频和文本的潜在表示
            return audio_latents, text_latents

        if return_similarities:
            # 计算相似度
            return einsum('i d, i d -> i', audio_latents, text_latents)

        if return_pairwise_similarities:
            # 计算成对相似度
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents) 
            return cosine_sim

        # 计算对比学习损失
        cl_loss = self.contrast(audio_latents, text_latents)

        if not exists(self.multi_layer_contrastive_learning):
            # 如果没有多层对比学习损失，则返回对比学习损失
            return cl_loss

        # 获取音频的多层嵌入向量
        audio_layers = audio_layers[self.audio_layers_indices]
        # 获取文本的多层嵌入向量
        text_layers = text_layers[self.text_layers_indices]

        # 计算多层对比学习损失
        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers = audio_layers, # 音频的多层嵌入向量
            text_layers = text_layers # 文本的多层嵌入向量
        )

        # 返回总损失
        return cl_loss + hierarchical_cl_loss 
    
