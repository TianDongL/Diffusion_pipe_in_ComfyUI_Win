import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.functional import quantize_nf4, dequantize_nf4

import logging

log = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class FusedLoRALayer(nn.Module):
    def __init__(self, in_features, fused_dim_list, rank=4, alpha=1):
        super().__init__()
        self.fused_dim_list = fused_dim_list
        self.lora_As = nn.Parameter(
            torch.zeros(rank * len(fused_dim_list), in_features)
        )
        self.lora_Bs = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, rank)) for dim in fused_dim_list]
        )
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_As, a=math.sqrt(5))

    def forward(self, x):
        As = F.linear(x, self.lora_As).chunk(len(self.fused_dim_list), dim=-1)
        Bs = []
        for i, lora_b in enumerate(self.lora_Bs):
            Bs.append(F.linear(As[i], lora_b))

        return torch.cat(Bs, dim=-1) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, fused_dim_list=None, rank=4, alpha=1):
        super().__init__()

        self.linear = linear
        self.linear.requires_grad_(False)
        if fused_dim_list:
            self.fused_dim_list = fused_dim_list
            self.is_fused_linear = True
            assert (
                sum(fused_dim_list) == linear.out_features
            ), f"sum of fused_dim_list {sum(fused_dim_list)} is not equal to linear.out_features {linear.out_features}"
            self.lora = FusedLoRALayer(linear.in_features, fused_dim_list, rank, alpha)
        else:
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.lora.to(device=linear.weight.device)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class Quantized8BitLinearWithLoRA(nn.Module):
    def __init__(
        self, linear, fused_dim_list=None, rank=4, alpha=1, quant=torch.float8_e4m3fn
    ):
        super().__init__()
        assert quant in {
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        }, "Unknown quantization"

        self.linear_weight = linear.weight.to(dtype=quant)
        self.linear_weight.detach()

        if linear.bias is not None:
            self.linear_bias = linear.bias.to(dtype=quant)
            self.linear_bias.detach()
        else:
            self.linear_bias = None

        if fused_dim_list:
            self.fused_dim_list = fused_dim_list
            self.is_fused_linear = True
            assert (
                sum(fused_dim_list) == linear.out_features
            ), f"sum of fused_dim_list {sum(fused_dim_list)} is not equal to linear.out_features {linear.out_features}"
            self.lora = FusedLoRALayer(linear.in_features, fused_dim_list, rank, alpha)
        else:
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

        self.lora.to(device=linear.weight.device)

    def forward(self, x):
        return F.linear(x, self.linear_weight, self.linear_bias) + self.lora(x)


class Quantized4BitLinearWithLoRA(nn.Module):
    def __init__(self, linear, fused_dim_list=None, rank=4, alpha=1):
        super().__init__()

        self.linear_weight = quantize_nf4(linear.weight)
        self.linear_weight[0].requires_grad_(False)

        if linear.bias is not None:
            self.linear_bias = quantize_nf4(linear.bias)
            self.linear_bias[0].requires_grad_(False)
        else:
            self.linear_bias = None

        if fused_dim_list:
            self.fused_dim_list = fused_dim_list
            self.is_fused_linear = True
            assert (
                sum(fused_dim_list) == linear.out_features
            ), f"sum of fused_dim_list {sum(fused_dim_list)} is not equal to linear.out_features {linear.out_features}"
            self.lora = FusedLoRALayer(linear.in_features, fused_dim_list, rank, alpha)
        else:
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

        self.lora.to(device=linear.weight.device)

    def forward(self, x):
        if self.linear_bias is not None:
            return F.linear(
                x,
                dequantize_nf4(*self.linear_weight),
                dequantize_nf4(*self.linear_bias),
            ) + self.lora(x)
        else:
            return F.linear(x, dequantize_nf4(*self.linear_weight), None) + self.lora(x)


class Quantized8bitLinear(nn.Module):
    def __init__(self, linear, quant=torch.float8_e4m3fn, device="cpu"):
        super().__init__()
        assert quant in {
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        }, "Unknown quantization"

        self.linear_weight = linear.weight.to(dtype=quant, device=device)
        if linear.bias is not None:
            self.linear_bias = linear.bias.to(dtype=quant, device=device)
        else:
            self.linear_bias = None

    def forward(self, x):
        return F.linear(x, self.linear_weight, self.linear_bias)


class Quantized4bitLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()

        self.linear_weight = quantize_nf4(linear.weight)
        self.linear_weight[0].requires_grad_(False)

        if linear.bias is not None:
            self.linear_bias = quantize_nf4(linear.bias)
            self.linear_bias[0].requires_grad_(False)
        else:
            self.linear_bias = None

    def forward(self, x):
        if self.linear_bias is not None:
            return F.linear(
                x,
                dequantize_nf4(*self.linear_weight),
                dequantize_nf4(*self.linear_bias),
            )
        else:
            return F.linear(x, dequantize_nf4(*self.linear_weight), None)


def swap_linear_recursive(
    model,
    replacement_module,
    exclude_keywords=None,
    fused_linear_patterns=None,
    **module_kwargs,
):
    # omitted layers
    if exclude_keywords is None:
        exclude_keywords = []

    # store regex pattern of fused layers
    if fused_linear_patterns is None:
        fused_linear_patterns = []

    def is_fused_linear(name):
        # check if the linear is fused linear like qkv linear fused together
        return any(re.match(pattern, name) for pattern, _ in fused_linear_patterns)

    def get_fused_dim_list(name):
        # how to partition the fused linear because some layer can be partitioned unevenly
        # ie: [256, 192, 256]
        for pattern, dim_list in fused_linear_patterns:
            if re.match(pattern, name):
                return dim_list
        return None

    def recursive_swap(module, parent_name=""):
        for name, child in module.named_children():
            current_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear) and not any(
                keyword in current_name for keyword in exclude_keywords
            ):
                if fused_linear_patterns != None and is_fused_linear(current_name):
                    fused_dim_list = get_fused_dim_list(current_name)
                    setattr(
                        module,
                        name,
                        replacement_module(
                            child, fused_dim_list=fused_dim_list, **module_kwargs
                        ),
                    )
                    print(
                        f"Replacing fused linear layer {current_name} with dimensions {fused_dim_list}"
                    )
                    log.info(
                        f"Replacing fused linear layer {current_name} with dimensions {fused_dim_list}"
                    )
                else:
                    setattr(module, name, replacement_module(child, **module_kwargs))
                    print(f"Replacing {current_name}")
                    log.info(f"Replacing {current_name}")
            else:
                recursive_swap(child, current_name)

    recursive_swap(model)


def swap_linear(model, replacement_module, exclude_keywords=None, **module_kwargs):
    if exclude_keywords is None:
        exclude_keywords = []

    # non recursive solution
    stack = [(model, "")]
    while stack:
        module, parent_name = stack.pop()
        for name, child in list(module.named_children()):
            current_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear) and not any(
                keyword in current_name for keyword in exclude_keywords
            ):
                # replace nn.Linear with another module
                setattr(module, name, replacement_module(child, **module_kwargs))
                log.info(f"Replacing {current_name}")
            else:
                stack.append((child, current_name))

    return model


def find_lora_params(model):
    lora_params = []
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            lora_params.append((n, p))
    return lora_params


def change_lora_scale(model, lora_instance, scale):
    def traverse_model(model, lora_instance, scale, path=""):
        for name, module in model.named_children():
            current_path = f"{path}.{name}" if path else name
            if isinstance(module, lora_instance):
                module.lora.scaling = scale
                log.info(f"changing {current_path} lora scale to {scale}")
            else:
                traverse_model(
                    model=module,
                    lora_instance=lora_instance,
                    scale=scale,
                    path=current_path,
                )

    traverse_model(model, lora_instance, scale, "")


def merge_lora_weights(model, replacement_module=LinearWithLoRA):
    for name, module in model.named_children():
        if isinstance(module, replacement_module):
            original_linear = module.linear
            lora_layer = module.lora

            if isinstance(lora_layer, LoRALayer):
                # Merge regular LoRA weights
                merged_weight = (
                    original_linear.weight
                    + (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
                )

                # Update the original linear layer's weight
                original_linear.weight.data.copy_(merged_weight)

            elif isinstance(lora_layer, FusedLoRALayer):
                # Merge Fused LoRA weights
                lora_As = lora_layer.lora_As.chunk(
                    len(lora_layer.fused_dim_list), dim=0
                )
                merged_weight = original_linear.weight.clone()

                start_idx = 0
                for i, (lora_A, lora_B) in enumerate(zip(lora_As, lora_layer.lora_Bs)):
                    end_idx = start_idx + lora_layer.fused_dim_list[i]
                    merged_weight[start_idx:end_idx] += (
                        lora_B @ lora_A
                    ) * lora_layer.scaling
                    start_idx = end_idx

                # Update the original linear layer's weight
                original_linear.weight.data.copy_(merged_weight)

            # Replace the LoRA module with the original linear layer
            setattr(model, name, original_linear)
        else:
            # Recursively apply to child modules
            merge_lora_weights(module, replacement_module)

    return model
