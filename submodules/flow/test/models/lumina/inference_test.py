import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from src.models.lumina.model import Lumina, Lumina_2b
from src.models.lumina.sampling import get_noise, get_schedule, denoise_cfg
from src.models.lumina.autoencoder import AutoEncoder, ae_params
from safetensors.torch import safe_open, save_file
from torchvision.utils import save_image
from transformers import Gemma2Model, Gemma2Config, GemmaTokenizerFast
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors

from torchvision.utils import make_grid, save_image
import torch.nn as nn


LUMINA_PATH = "/media/lodestone/bulk_storage/model/lumina.sft"
VAE_PATH = "models/flux/ae.safetensors"
GEMMA_PATH = "/media/lodestone/bulk_storage_2/models/gemma-2-2b"
GEMMA_CONFIG_PATH = "/media/lodestone/bulk_storage_2/models/gemma-2-2b/config.json"
GEMMA_TOKENIZER_PATH = "/media/lodestone/bulk_storage_2/models/gemma-2-2b"
DEVICE = "cuda:1"
GEMMA_DEVICE = "cuda:0"

# load model
with torch.device("meta"):
    model = Lumina_2b()
model.load_state_dict(load_safetensors(LUMINA_PATH),  assign=True)
model.to(DEVICE)
model.set_use_compiled()
# load ae
with torch.device("meta"):
    ae = AutoEncoder(ae_params)
ae.load_state_dict(load_safetensors(VAE_PATH), assign=True)
ae.to(DEVICE)

# load t5
gemma_tokenizer = GemmaTokenizerFast.from_pretrained(GEMMA_TOKENIZER_PATH)
gemma_tokenizer.padding_side = "right"


gemma = Gemma2Model.from_pretrained(GEMMA_PATH, torch_dtype=torch.bfloat16)

gemma.to(DEVICE)
def cast_linear(module, dtype):
    """
    Recursively cast all nn.Linear layers in the model to bfloat16.
    """
    for name, child in module.named_children():
        # If the child module is nn.Linear, cast it to bf16
        if isinstance(child, nn.Linear):
            child.to(dtype)
        else:
            # Recursively apply to child modules
            cast_linear(child, dtype)
cast_linear(gemma, torch.float8_e4m3fn)
gemma.eval()


torch.cuda.empty_cache()

#############################################################################
# test inference
SEED = 10
WIDTH = 1024
HEIGHT = 1024
STEPS = 20
GUIDANCE = 1
CFG = 5
FIRST_N_STEPS_WITHOUT_CFG = 0
# DEVICE = 0
PROMPT = [
    "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> a cute cat sat on a mat while receiving a head pat from his owner called Matt",
    "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> baked potato, on the space floating orbiting around the earth",
]
NEGATIVE_PROMPT = [
    "",
    "",
]

GEMMA_MAX_LENGTH = 256
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # init random noise
        torch.manual_seed(SEED)
        noise = get_noise(len(PROMPT), HEIGHT, WIDTH, DEVICE, torch.bfloat16, SEED)
        noise = noise.to(model.device)

        timesteps = get_schedule(STEPS, WIDTH //16 * HEIGHT//16)

        text_inputs = gemma_tokenizer(
            PROMPT,
            padding="max_length",
            # padding=True,
            # pad_to_multiple_of=8,
            max_length=GEMMA_MAX_LENGTH,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).to(gemma.device)

        gemma_embed = gemma(text_inputs.input_ids, text_inputs.attention_mask,output_hidden_states=True,).hidden_states[-2]

        text_inputs_neg = gemma_tokenizer(
            NEGATIVE_PROMPT,
            padding="max_length",
            # padding=True,
            # pad_to_multiple_of=8,
            max_length=GEMMA_MAX_LENGTH,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).to(gemma.device)

        gemma_embed_neg = gemma(text_inputs_neg.input_ids, text_inputs_neg.attention_mask, output_hidden_states=True,).hidden_states[-2]

        gemma.to("cpu")
        torch.cuda.empty_cache()
        latent_cfg = denoise_cfg(
            model,
            noise,
            gemma_embed,
            text_inputs.attention_mask,
            gemma_embed_neg,
            text_inputs_neg.attention_mask,
            timesteps,
            CFG,
            FIRST_N_STEPS_WITHOUT_CFG,
        )
        model.to("cpu")

        torch.cuda.empty_cache()
        output_image = ae.decode(latent_cfg)

grid = make_grid(output_image.clip(-1,1), nrow=2, padding=2, normalize=True)

# Option 1: Save directly with save_image
save_image(grid, "grid_image6.png")

print()
