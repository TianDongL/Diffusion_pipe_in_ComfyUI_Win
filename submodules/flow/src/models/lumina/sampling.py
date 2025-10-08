import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Lumina
from tqdm import tqdm


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor):
    # flipped because lumina use t=0 as full noise and t=1 as full image
    t = 1 - t
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    t = 1 - t
    return t


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # flipped because lumina use t=0 as full noise and t=1 as full image
    # extra step for one
    timesteps = torch.linspace(0, 1, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Lumina,
    # model input
    img: Tensor,
    txt: Tensor,
    mask: Tensor,
    # sampling parameters
    timesteps: list[float],
):

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            x=img,
            t=t_vec,
            cap_feats=txt,
            cap_mask=mask,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def denoise_cfg(
    model: Lumina,
    # model input
    img: Tensor,
    # positive guidance
    txt: Tensor,
    mask: Tensor,
    # negative guidance
    neg_txt: Tensor,
    neg_mask: Tensor,
    # sampling parameters
    timesteps: list[float],
    cfg: float = 2.0,
    first_n_steps_without_cfg: int = 4,
):
    step_count = 0
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            x=img,
            t=t_vec,
            cap_feats=txt,
            cap_mask=mask,
        )
        # disable cfg for x steps before using cfg
        if step_count < first_n_steps_without_cfg or first_n_steps_without_cfg == -1:
            img = img.to(pred) + (t_prev - t_curr) * pred
        else:
            pred_neg = model(
                x=img,
                t=t_vec,
                cap_feats=neg_txt,
                cap_mask=neg_mask,
            )

            pred_cfg = pred_neg + (pred - pred_neg) * cfg

            img = img + (t_prev - t_curr) * pred_cfg

        step_count += 1

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
