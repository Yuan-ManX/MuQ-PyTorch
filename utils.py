from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    """
    检查值是否存在。

    参数:
        val (Optional[Any]): 要检查的值。

    返回:
        bool: 如果值存在且不为 None，则返回 True，否则返回 False。
    """
    return val is not None


def first(it):
    """
    返回可迭代对象中的第一个元素。

    参数:
        it (Iterable[T]): 要迭代的可迭代对象。

    返回:
        T: 可迭代对象中的第一个元素。
    """
    return it[0]


def default(val, d):
    """
    如果值存在，则返回该值；否则，返回默认值。

    参数:
        val (Optional[T]): 要检查的值。
        d (T): 默认值。

    返回:
        T: 如果 val 存在，则返回 val；否则，返回 d。
    """
    return val if exists(val) else d


def round_down_nearest_multiple(n, divisor):
    """
    将一个数向下取整到最接近的指定除数的倍数。

    参数:
        n (int): 要取整的数。
        divisor (int): 除数。

    返回:
        int: 向下取整到最接近的指定除数的倍数的结果。
    """
    return n // divisor * divisor


def Sequential(*modules):
    """
    创建一个顺序容器，按顺序执行模块。

    该函数类似于 PyTorch 的 nn.Sequential，但会过滤掉所有为 None 的模块。

    参数:
        *modules (Optional[nn.Module]): 要包含在顺序容器中的模块，可以有多个。

    返回:
        nn.Sequential: 包含过滤后模块的顺序容器。
    """
    return nn.Sequential(*filter(exists, modules))


def once(fn):
    """
    确保一个函数只被调用一次。

    参数:
        fn (Callable[[T], Any]): 要包装的函数。

    返回:
        Callable[[T], Optional[Any]]: 包装后的函数，第二次调用时不会执行原函数。
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


# tensor functions

def log(t, eps = 1e-20):
    """
    计算张量的对数。

    为了避免数值不稳定，对输入张量进行最小值裁剪。

    参数:
        t (torch.Tensor): 输入张量。
        eps (float, 可选): 最小值裁剪值，默认为 1e-20。

    返回:
        torch.Tensor: 输入张量的对数。
    """
    return torch.log(t.clamp(min = eps))


def l2norm(t):
    """
    对张量进行 L2 归一化。

    参数:
        t (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: L2 归一化后的张量。
    """
    return F.normalize(t, p = 2, dim = -1)


def frozen_params(model:nn.Module):
    """
    冻结模型的所有参数，防止参数更新。

    参数:
        model (nn.Module): 要冻结参数的模型。
    """
    for param in model.parameters():
        param.requires_grad = False
        
