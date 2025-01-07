from typing import List, Optional
from dataclasses import dataclass, field
import os

from torch.nn.parallel.distributed import DistributedDataParallel
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from easydict import EasyDict

from mulan import MuLanModel
from audio import AudioSpectrogramTransformerPretrained
from text import TextTransformerPretrained
from utils import exists, frozen_params


@dataclass
class MuLanConfig:
    """
    MuLan 模型的配置类。

    该类定义了 MuLan 模型的各种配置参数，用于控制模型的行为和性能。
    """
    sr:int = field(default=24000)
    clip_secs:float = field(default=10) # 每个音频片段的持续时间（秒），默认为 10 秒
    dim_latent:int = field(default=512) # 潜在空间的维度，默认为 512
    decoupled_contrastive_learning:bool = field(default=True) # 是否使用解耦对比学习，默认为 True
    hierarchical_contrastive_loss:bool = field(default=False) # 是否使用分层对比损失，默认为 False
    hierarchical_contrastive_loss_layers:Optional[List] = field(default=None) # 分层对比损失中使用的层列表，默认为 None
    sigmoid_contrastive_loss:bool = field(default=False) # 是否使用 sigmoid 对比损失，默认为 False
    rank_contrast:bool = field(default=True) # 是否使用排序对比，默认为 True


@dataclass
class AudioTransformerConfig:
    """
    音频 Transformer 的配置类。

    该类定义了音频 Transformer 的各种配置参数，用于控制音频 Transformer 的结构和行为。
    """
    dim:int = field(default=768) # Transformer 模型的维度，默认为 768
    tf_depth:int = field(default=8) # Transformer 模型的深度（即层数），默认为 8
    heads:int = field(default=8) # 多头自注意力中的头数，默认为 8
    dim_head:int = field(default=64) # 每个头的维度，默认为 64
    attn_dropout:float = field(default=0.) # 自注意力层的 dropout 概率，默认为 0
    ff_dropout:float = field(default=0.) # 前馈层的 dropout 概率，默认为 0
    ff_mult:int = field(default=4) # 前馈层中隐藏层的维度倍数，默认为 4


@dataclass
class TextTransformerConfig:
    """
    文本 Transformer 的配置类。

    该类定义了文本 Transformer 的各种配置参数，用于控制文本 Transformer 的结构和行为。
    """
    dim:int = field(default=768) # Transformer 模型的维度，默认为 768
    tf_depth:int = field(default=8) # Transformer 模型的深度（即层数），默认为 8
    max_seq_len:int = field(default=1024) # 最大序列长度，默认为 1024
    dim_head:int = field(default=64) # 每个头的维度，默认为 64
    heads:int = field(default=8) # 多头自注意力中的头数，默认为 8
    attn_dropout:float = field(default=0.) # 自注意力层的 dropout 概率，默认为 0
    ff_dropout:float = field(default=0.) # 前馈层的 dropout 概率，默认为 0
    ff_mult:int = field(default=4) # 前馈层中隐藏层的维度倍数，默认为 4


@dataclass
class ModalModelConfig:
    """
    模态模型（Modal Model）的配置类。

    该类定义了模态模型的各种配置参数，用于控制模态模型的配置和行为。
    """
    name:str = field(default='') # 模型名称，默认为空字符串
    model_dim: Optional[int] = field(default=None) # 模型维度，可选，默认为 None
    use_layer_idx: int = field(default=-1) # 要使用的层索引，默认为 -1（表示使用最后一层）


@dataclass
class MuQMuLanConfig:
    """
    MuQMuLan 模型的配置类。

    该类聚合了 MuLan 模型及其相关组件的配置参数，用于全面配置整个模型系统。
    """
    mulan: MuLanConfig # MuLan 模型的配置
    audio_model: ModalModelConfig # 音频模型的配置
    text_model: ModalModelConfig # 文本模型的配置 
    audio_transformer: AudioTransformerConfig # 音频 Transformer 的配置
    text_transformer: TextTransformerConfig # 文本 Transformer 的配置


class MuQMuLan(nn.Module, PyTorchModelHubMixin):
    """
    MuQMuLan 模型类。

    该类实现了 MuQMuLan 模型，包括模型的初始化、从预训练模型加载以及模型冻结等功能。
    """
    def __init__(self, config: MuQMuLanConfig, hf_hub_cache_dir=None):
        """
        初始化 MuQMuLan 模型。

        参数:
            config (MuQMuLanConfig): 模型的配置参数。
            hf_hub_cache_dir (Optional[str]): Hugging Face 模型缓存目录，可选。
        """
        super().__init__()
        # 将配置参数转换为 EasyDict 对象
        config = self._to_obj(config)
        self.config = config
        self.mulan = self.create_MuLan_from_config(config, hf_hub_cache_dir)
        self.sr = config.mulan.sr
        self.clip_secs = config.mulan.clip_secs
    
    def _to_obj(self, config):
        """
        将配置参数转换为 EasyDict 对象。

        参数:
            config: 配置参数，可以是 MuQMuLanConfig 对象或字典。

        返回:
            EasyDict: 转换后的配置参数。
        """
        if isinstance(config, MuQMuLanConfig):
            # 如果配置参数是 MuQMuLanConfig 对象，则转换为 EasyDict 对象
            config = EasyDict(
                mulan = config.mulan,
                audio_model = config.audio_model,
                text_model = config.text_model,
                audio_transformer = config.audio_transformer,
                text_transformer = config.text_transformer,
            )
        else:
            # 否则，直接转换为 EasyDict 对象
            config = EasyDict(config)
        return config
    
    @classmethod
    def from_pretrained(cls, *args, cache_dir=None, **kwargs):
        """
        从预训练模型加载 MuQMuLan 模型。

        参数:
            *args: 位置参数。
            cache_dir (Optional[str]): 模型缓存目录，可选。
            **kwargs: 关键字参数。

        返回:
            MuQMuLan: 加载的 MuQMuLan 模型实例。
        """
        # 设置 Hugging Face 模型缓存目录
        kwargs['hf_hub_cache_dir'] = cache_dir
        return super().from_pretrained(*args, cache_dir=cache_dir, **kwargs)


    @classmethod
    def create_MuLan_from_config(cls, config:MuQMuLanConfig, hf_hub_cache_dir=None) -> MuLanModel:
        """
        根据配置参数创建 MuLan 模型。

        参数:
            config (MuQMuLanConfig): 模型的配置参数。
            hf_hub_cache_dir (Optional[str]): Hugging Face 模型缓存目录，可选。

        返回:
            MuLanModel: 创建的 MuLan 模型实例。
        """
        # 创建音频 Transformer 模型
        audio_transformer = AudioSpectrogramTransformerPretrained(
            model_name = config.audio_model.name, 
            model_dim = config.audio_model.model_dim,
            use_layer_idx = config.audio_model.use_layer_idx,
            **config.audio_transformer,
            frozen_pretrained = False, # 是否冻结预训练模型
            hf_hub_cache_dir = hf_hub_cache_dir,
        )
        # 创建文本 Transformer 模型
        text_transformer = TextTransformerPretrained(
            model_name = config.text_model.name, 
            model_dim = config.text_model.model_dim,
            **config.text_transformer,
            frozen_pretrained = False, # 是否冻结预训练模型
            hf_hub_cache_dir = hf_hub_cache_dir,
        )

        # 创建 MuLan 模型
        mulan = MuLanModel(
            audio_transformer = audio_transformer, # 音频 Transformer 模型
            text_transformer = text_transformer, # 文本 Transformer 模型
            **config.mulan 
        )

        return mulan
    
    def frozen(self):
        """
        冻结模型参数。
        """
        frozen_params(self)

    @property
    def device(self):
        """
        获取模型所在的设备。

        返回:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device
    
    @property
    def mulan_module(self):
        """
        获取 MuLan 模型的模块。

        如果 MuLan 模型是分布式数据并行模型，则返回其内部的模块。
        否则，直接返回 MuLan 模型。

        返回:
            MuLanModel: MuLan 模型的模块。
        """
        if isinstance(self.mulan, DistributedDataParallel):
            # 如果是分布式数据并行模型，则返回其内部的模块
            return self.mulan.module
        else:
            # 否则，直接返回 MuLan 模型
            return self.mulan
    
    def forward(self,
        wavs: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        *,
        parallel_processing = False, 
    ) -> torch.Tensor:    
        """
        提取音频或文本的特征，输入可以是音频或文本批次。
        注意，如果音频长度超过 10 秒，它将被裁剪为多个片段，并返回平均的潜在表示。
        参数 `parallel_processing` 用于控制是否使用并行处理。
        如果设置为 True，则使用并行处理进行特征提取，速度更快但占用更多 GPU 内存。
        如果设置为 False（默认值），则使用串行处理进行特征提取，速度较慢但内存友好。

        参数:
            wavs (Optional[torch.Tensor]): 音频波形张量。默认为 None。
            texts (Optional[List[str]]): 文本字符串列表。默认为 None。
            parallel_processing (bool): 是否使用并行处理。默认为 False。

        返回:
            torch.Tensor: 音频或文本输入的潜在表示。

        异常:
            AssertionError: 如果同时提供了 wavs 和 texts，或者两者都没有提供，则抛出断言错误。

        注意:
            - 必须提供 wavs 或 texts 中的一个，但不能同时提供。
            - 如果提供了 wavs，则调用 extract_audio_latents 方法处理音频。
            - 如果提供了 texts，则调用 extract_text_latents 方法处理文本。
        """
        assert exists(wavs) ^ exists(texts), "Please provide either wavs or texts, but not both"
        
        if exists(wavs):
            # 如果提供了音频，则提取音频特征
            return self.extract_audio_latents(wavs = wavs, parallel_processing = parallel_processing)
        else: 
            # 如果提供了文本，则提取文本特征
            return self.extract_text_latents(texts = texts)
    
    def calc_similarity(self, audio_latents: torch.Tensor, text_latents: torch.Tensor) -> torch.Tensor:
        """
        计算音频和文本潜在表示之间的点积相似度。
        它支持各种维度的输入张量（包含或不包含批次维度）的音频和文本。

        注意:
            这个函数的效果基本上等同于点积。
            mulan.calc_similarity(lat_a, lat_t) <==> einsum('i d, j d -> i j', lat_a, lat_t)

        参数:
            audio_latents (torch.Tensor): 音频的潜在表示。
            text_latents (torch.Tensor): 文本的潜在表示。

        返回:
            torch.Tensor: 音频和文本潜在表示之间的相似度分数。

        """
        # 获取音频和文本张量的维度
        dim_a, dim_t = len(audio_latents.shape), len(text_latents.shape)
        
        if dim_a == 2 and dim_t == 2:
            # 如果都是二维张量，则计算批量点积
            return einsum('i d, j d -> i j', audio_latents, text_latents)
        elif dim_a == 1 and dim_t == 1:
            # 如果都是一维张量，则计算点积
            return torch.dot(audio_latents, text_latents)
        elif dim_a == 2 and dim_t == 1:
            # 如果音频是二维，文本是一维，则计算批量点积
            return einsum('i d, d -> i', audio_latents, text_latents)
        elif dim_a == 1 and dim_t == 2:
            # 如果音频是一维，文本是二维，则计算批量点积
            return einsum('d, j d -> j', audio_latents, text_latents)
        
        raise RuntimeError(f"Invalid dimensions: audio {dim_a}, text {dim_t}")
        
    
    def extract_audio_latents(self, wavs:torch.Tensor, *, parallel_processing = False) -> torch.Tensor:
        """
        从音频波形中提取潜在表示。

        该函数处理一批音频波形，并提取它们的潜在表示。
        它支持并行处理以加快计算速度，但会占用更多 GPU 内存。

        参数:
            wavs (torch.Tensor): 一批音频波形张量。
            parallel_processing (bool): 启用并行处理的标志。默认为 False。

        返回:
            torch.Tensor: 包含输入音频波形潜在表示的张量。
        """
        # 初始化音频潜在表示列表
        audio_latents = []

        def audio_to_latent(wav):
            # 调用 MuLan 模块的 get_audio_latents 方法提取音频潜在表示
            return self.mulan_module.get_audio_latents(wav)
        for wav in wavs:
            # 初始化每个音频样本的剪辑列表
            wav_tensors = []
            if isinstance(wav, torch.Tensor):
                # 将音频波形分割为多个剪辑
                wav_tensors = self._get_all_clips(wav)
            else: 
                raise TypeError('wavs must be a Tensor')
            
            if parallel_processing:
                wav_tensors = wav_tensors.to(self.device)
                # 提取音频潜在表示
                audio_latent = audio_to_latent(wav_tensors)
                # 对批次维度求平均
                audio_latent = audio_latent.mean(dim=0)
            else:  
                # 重塑张量以适应批量处理
                wav_tensors = rearrange(wav_tensors, "i j -> i 1 j")
                # 初始化音频潜在表示列表
                audio_latent = []
                for wav_tensor in wav_tensors:
                    # 提取每个剪辑的潜在表示并去除批次维度
                    audio_latent.append(audio_to_latent(wav_tensor).squeeze(0))
                    # 删除临时变量以释放内存
                    del wav_tensor
                # 堆叠所有潜在表示
                audio_latent = torch.stack(audio_latent, dim=0)
                # 对批次维度求平均并移动到设备      
                audio_latent = audio_latent.mean(dim=0).to(self.device)      
            
             # 将每个音频样本的潜在表示添加到列表中
            audio_latents.append(audio_latent)
        # 堆叠所有音频样本的潜在表示
        audio_latents = torch.stack(audio_latents, dim=0)
        # 返回音频潜在表示张量
        return audio_latents

    def extract_text_latents(self, texts: List[str]) -> torch.Tensor:
        """
        从文本输入中提取潜在表示。

        该函数处理一个文本字符串列表，并使用 MuLan 模型的文本塔提取它们的潜在表示。

        参数:
            texts (List[str]): 要处理的文本字符串列表。

        返回:
            torch.Tensor: 包含输入文本潜在表示的张量。
        """
        # 调用 MuLan 模块的 get_text_latents 方法提取文本潜在表示
        return self.mulan_module.get_text_latents(raw_texts=texts)
    
    def _get_all_clips(self, audio):
        """
        将音频波形分割为多个剪辑。

        该函数将输入的音频波形分割为多个固定长度的剪辑，每个剪辑的持续时间为 `clip_secs` 秒。

        参数:
            audio (torch.Tensor): 输入的音频波形张量。

        返回:
            List[torch.Tensor]: 包含所有剪辑的列表。
        """
        origin_length = len(audio)  # 获取音频的总长度
        accum_length = 0  # 初始化累积长度
        delta = self.sr * self.clip_secs  # 计算每个剪辑的长度
        audio_clips = []  # 初始化剪辑列表

        while accum_length + delta <= origin_length:
            # 分割出当前剪辑
            clip = audio[accum_length:accum_length + delta] 
            # 添加到剪辑列表中
            audio_clips.append(clip)
            # 更新累积长度
            accum_length += delta

        if accum_length < origin_length:
            # 如果最后剩余的音频长度不足一个剪辑，则将剩余部分与开头的部分拼接
            audio_clips.append(torch.cat([audio[accum_length:], audio[0:delta - (origin_length - accum_length)]]))

        # 将所有剪辑堆叠成一个张量
        return torch.stack(audio_clips, dim=0)
