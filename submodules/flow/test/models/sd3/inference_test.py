import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from safetensors import safe_open

from src.models.sd3.sd3_impls import ModelSamplingDiscreteFlow
from src.models.sd3.other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from src.models.sd3.sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)



CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "cfg": 5.0,
        "steps": 50,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large": {
        "shift": 3.0,
        "cfg": 4.5,
        "steps": 40,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large_turbo": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
}



def load_into(f, model, prefix, device, dtype=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


class ClipG:
    def __init__(self, model, device="cpu", dtype=torch.bfloat16):
        with safe_open(model, framework="pt", device=device) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=dtype)
            self.model.to(dtype)
            load_into(f, self.model.transformer, "", device, dtype)


class ClipL:
    def __init__(self, model, device="cpu", dtype=torch.bfloat16):
        with safe_open(model, framework="pt", device=device) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device=device,
                dtype=dtype,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            self.model.to(device, dtype)
            load_into(f, self.model.transformer, "", device, dtype)


class T5XXL:
    def __init__(self, model, device="cpu", dtype=torch.bfloat16):
        with safe_open(model, framework="pt", device=device) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            self.model.to(device, dtype)
            load_into(f, self.model.transformer, "", device, dtype)


class SD3:
    def __init__(self, model, shift, device="cpu", dtype=torch.bfloat16, verbose=False):
        with safe_open(model, framework="pt", device=device) as f:
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device=device,
                dtype=dtype,
                verbose=verbose,
            )
            self.model.to(device, dtype)
            load_into(f, self.model, "model.", device, dtype)


class VAE:
    def __init__(self, model, device="cpu", dtype=torch.bfloat16):
        with safe_open(model, framework="pt", device=device) as f:
            self.model = SDVAE(device=device, dtype=dtype).eval()
            self.model.to(device, dtype)
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, device, dtype)



MODEL = "models/sd3.5_med/sd3.5_medium.safetensors"
CLIP_L = "models/sd3.5_med/text_encoders/clip_l.safetensors"
CLIP_G = "models/sd3.5_med/text_encoders/clip_g.safetensors"
T5_XXL = "models/sd3.5_med/text_encoders/t5xxl_fp16.safetensors"

with torch.no_grad():
    tokenizer = SD3Tokenizer()
    print("Loading OpenAI CLIP L...")
    clip_l = ClipL(CLIP_L, device="cuda:0")
    print("Loading OpenCLIP bigG...")
    clip_g = ClipG(CLIP_G, device="cuda:0")
    print("Loading Google T5-v1-XXL...")
    t5xxl = T5XXL(T5_XXL, device="cuda:0")
    print(f"Loading SD3 model {os.path.basename(MODEL)}...")
    sd3 = SD3(MODEL, CONFIGS["sd3.5_large"]["shift"], verbose=True, device="cuda:1")
    print("Loading VAE model...")
    vae = VAE(MODEL, device="cuda:1")
    print("Models loaded.")
    # clip_l.model.to("cuda:0")
    # clip_g.model.to("cuda:0")
    # t5xxl.model.to("cuda:0")
    def get_cond(prompt):
        print("Encode prompt...")
        tokens = tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    a = get_cond("cat")

    # 1, 16, 128, 128
    # 1 (one d for t)
    # 1, 154, 4096
    # 1, 2048
    sd3.model()
print()