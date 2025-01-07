import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.distributed as dist
from einops import rearrange


def exists(val):
    """
    检查值是否存在。

    参数:
        val (Optional[Any]): 要检查的值。

    返回:
        bool: 如果值存在且不为 None，则返回 True，否则返回 False。
    """
    return val is not None


def pad_dim_to(t: torch.Tensor, target_size: int, dim: int = 0) -> torch.Tensor:
    """
    对张量在指定维度上进行填充，使其达到目标大小。

    参数:
        t (torch.Tensor): 要填充的张量。
        target_size (int): 目标大小。
        dim (int, 可选): 要填充的维度，默认为 0。

    返回:
        torch.Tensor: 填充后的张量。
    """
    current_size = t.shape[dim]  # 获取当前张量在指定维度的大小
    pad_size = target_size - current_size  # 计算需要填充的大小
    if pad_size <= 0:
        return t  # 如果不需要填充，则直接返回原张量

    pad = [0] * (2 * t.dim())  # 初始化填充列表
    pad[2 * dim] = pad_size  # 设置在指定维度的填充大小
    return F.pad(t, pad, mode='constant', value=0)  # 对张量进行填充，填充值为 0


# distributed helpers
# 分布式

def all_gather_same_dim(t):
    """
    在所有进程中收集具有相同维度的张量。

    该函数在所有分布式进程中收集张量，并返回每个进程的张量列表。
    假设所有进程中的张量具有相同的维度。

    参数:
        t (torch.Tensor): 要收集的张量。

    返回:
        List[torch.Tensor]: 包含所有进程中张量的列表。
    """
    # 获取分布式训练中的进程数
    world_size = dist.get_world_size()
    # 为每个进程创建一个空张量，形状与输入张量相同
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    # 在所有进程中收集张量
    dist.all_gather(gathered_tensors, t)
    # 返回收集到的张量列表
    return gathered_tensors


def all_gather_variable_dim(t, dim = 0, sizes = None):
    """
    在所有进程中收集具有可变维度的张量。

    该函数在所有分布式进程中收集张量，并处理不同进程中的张量可能具有不同维度的情况。
    返回一个连接后的张量和每个进程的张量大小。

    参数:
        t (torch.Tensor): 要收集的张量。
        dim (int, 可选): 要连接的维度，默认为 0。
        sizes (Optional[List[int]], 可选): 每个进程的张量大小列表，如果为 None，则自动收集。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 连接后的张量和每个进程的张量大小。
    """
    # 获取设备、当前进程排名和进程数
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        # 获取当前张量在指定维度的大小
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        # 收集所有进程中的大小
        sizes = all_gather_same_dim(size)
        # 将大小列表堆叠成一个张量
        sizes = torch.stack(sizes)

    if torch.unique(sizes).numel() == 1:
        # 如果所有进程中的大小相同，则使用 all_gather_same_dim 进行收集
        gathered_tensors = all_gather_same_dim(t)
        # 连接连接后的张量和大小
        return torch.cat(gathered_tensors, dim = dim), sizes
    
    # 获取所有进程中指定维度的最大大小
    max_size = sizes.amax().item()
    
    # 对张量进行填充，使其在指定维度上达到最大大小
    padded_t = pad_dim_to(t, max_size, dim = dim)
    # 收集填充后的张量
    gathered_tensors = all_gather_same_dim(padded_t)

    # 连接所有收集到的张量
    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # 创建一个序列张量 [0, 1, 2, ..., max_size-1]
    seq = torch.arange(max_size, device = device)

    # 创建掩码，标记哪些位置是有效的
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    # 重塑掩码形状
    mask = rearrange(mask, 'i j -> (i j)')
    # 创建一个序列张量 [0, 1, 2, ..., N-1]
    seq = torch.arange(mask.shape[-1], device = device)
    # 获取有效的索引
    indices = seq[mask]
    
    # 根据有效索引选择张量元素
    gathered_tensor = gathered_tensor.index_select(dim, indices)

    # 返回连接后的张量和大小
    return gathered_tensor, sizes


class AllGatherFunction(Function):
    """
    AllGatherFunction 类。

    该类实现了自定义的 all_gather 操作，用于在分布式训练中收集张量，并处理梯度聚合。
    """
    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        """
        前向传播方法，执行 all_gather 操作。

        参数:
            ctx (context): 上下文对象，用于保存反向传播所需的信息。
            x (torch.Tensor): 输入张量。
            dim (int): 要连接的维度。
            sizes (Optional[List[int]]): 每个进程的张量大小列表，如果为 None，则自动收集。
            all_reduce_grads (bool): 是否在反向传播时对梯度进行 all_reduce 操作。

        返回:
            Tuple[torch.Tensor, List[int]]: 收集后的张量和每个进程的张量大小。
        """
        # 使用 all_gather_variable_dim 函数收集张量，并获取每个进程的张量大小
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        # 将 dim 和 all_reduce_grads 保存到上下文对象中
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        # 将 batch_sizes 转换为列表并保存到上下文对象中
        ctx.batch_sizes = batch_sizes.tolist()
        # 返回收集后的张量和 batch_sizes
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        """
        反向传播方法，分配梯度并执行 all_reduce 操作（如果需要）。

        参数:
            ctx (context): 上下文对象，包含前向传播保存的信息。
            grads (torch.Tensor): 输入张量的梯度。
            _ (Any): 第二个输出梯度的占位符（未使用）。

        返回:
            Tuple[torch.Tensor, None, None, None]: 返回分配后的梯度，以及三个 None 值。
        """
        # 获取每个进程的张量大小和当前进程的排名
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        # 如果需要，对梯度进行 all_reduce 操作
        if ctx.all_reduce_grads:
            dist.all_reduce(grads)

        # 将梯度按照 batch_sizes 分割，每个进程只保留自己的梯度部分
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        # 返回当前进程的梯度，以及三个 None 值（因为输入有三个额外的参数，但不需要梯度）
        return grads_by_rank[rank], None, None, None


class AllGather(nn.Module):
    """
    AllGather 类。

    该类封装了 AllGatherFunction，提供一个易于使用的接口，用于在分布式训练中收集张量。
    """
    def __init__(
        self,
        dim,
        *,
        all_reduce_grads = False
    ):
        """
        初始化 AllGather 模块。

        参数:
            dim (int): 要连接的维度。
            all_reduce_grads (bool, 可选): 是否在反向传播时对梯度进行 all_reduce 操作，默认为 False。
        """
        super().__init__()
        # 要连接的维度
        self.dim = dim
        # 是否对梯度进行 all_reduce 操作
        self.all_reduce_grads = all_reduce_grads
        # 检查是否已初始化分布式训练，以及进程数是否大于 1
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def forward(
        self,
        x,
        sizes = None
    ):
        """
        前向传播方法，执行 all_gather 操作。

        参数:
            x (torch.Tensor): 输入张量。
            sizes (Optional[List[int]], 可选): 每个进程的张量大小列表，如果为 None，则自动收集。

        返回:
            Tuple[torch.Tensor, List[int]]: 收集后的张量和每个进程的张量大小。
        """
        # 调用 AllGatherFunction 的 apply 方法执行 all_gather 操作
        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)
