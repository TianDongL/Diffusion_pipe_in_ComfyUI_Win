import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import vae_flatten, prepare_latent_image_ids, vae_unflatten
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from safetensors.torch import safe_open, save_file
from torchvision.utils import save_image
from transformers import T5Tokenizer
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import src.lora_and_quant as lora_and_quant

from torchvision.utils import make_grid, save_image


MODEL_PATH = "models/flux/flux-dev.safetensors"
GUIDANCE_PATH = "models/flux/universal_modulator_v2.1.safetensors"
CHROMA_PATH = "models/flux/FLUX.1-schnell/chroma-8.9b.safetensors"
VAE_PATH = "models/flux/ae.safetensors"
T5_PATH = "models/flux/text_encoder_2"
T5_CONFIG_PATH = "models/flux/text_encoder_2/config.json"
T5_TOKENIZER_PATH = "models/flux/tokenizer_2"
DEVICE = "cuda:1"
T5_DEVICE = "cuda:0"

# load model
with torch.device("meta"):
    model = Chroma(chroma_params)
# state_dict_backbone = load_selected_keys(
#     MODEL_PATH, ["mod", "time_in", "guidance_in", "vector_in"]
# )
# guidance_backbone = load_safetensors(GUIDANCE_PATH)
# model.load_state_dict(state_dict_backbone, strict=False, assign=True)
# model.distilled_guidance_layer.load_state_dict(guidance_backbone, assign=True)
# save_file(model.state_dict(), "models/flux/chroma_init.safetensors")
model.load_state_dict(load_safetensors(CHROMA_PATH),  assign=True)
model.to(DEVICE)
# load ae
with torch.device("meta"):
    ae = AutoEncoder(ae_params)
ae.load_state_dict(load_safetensors(VAE_PATH), assign=True)
ae.to(DEVICE)

# load t5
t5_tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_PATH)
t5_config = T5Config.from_json_file(T5_CONFIG_PATH)
with torch.device("meta"):
    t5 = T5EncoderModel(t5_config)
t5.load_state_dict(replace_keys(load_file_multipart(T5_PATH)), assign=True)
# quantize t5
lora_and_quant.swap_linear_recursive(
    t5, lora_and_quant.Quantized8bitLinear, device=T5_DEVICE
)
t5.to(T5_DEVICE)
t5.eval()


#############################################################################
# test inference
SEED = 0
WIDTH = 512
HEIGHT = 512
STEPS = 20
GUIDANCE = 3
CFG = 1
FIRST_N_STEPS_WITHOUT_CFG = -1
DEVICE = model.device
PROMPT = [
    "a cute cat sat on a mat while receiving a head pat from his owner called Matt",
    "baked potato, on the space floating orbiting around the earth",
]
T5_MAX_LENGTH = 512
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # init random noise
        torch.manual_seed(SEED)
        noise = get_noise(len(PROMPT), HEIGHT, WIDTH, DEVICE, torch.bfloat16, 0)
        noise, shape = vae_flatten(noise)
        noise = noise.to(model.device)
        n, c, h, w = shape
        image_pos_id = prepare_latent_image_ids(n, h, w).to(model.device)

        timesteps = get_schedule(STEPS, noise.shape[1])

        text_inputs = t5_tokenizer(
            PROMPT,
            padding="max_length",
            max_length=T5_MAX_LENGTH,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).to(t5.device)

        t5_embed = t5(text_inputs.input_ids).to(model.device)

        text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=model.device)

        latent_cfg = denoise_cfg(
            model,
            noise,
            image_pos_id,
            t5_embed,
            torch.zeros_like(t5_embed),
            text_ids,
            timesteps,
            GUIDANCE,
            CFG,
            FIRST_N_STEPS_WITHOUT_CFG,
        )

        output_image = ae.decode(vae_unflatten(latent_cfg, shape))

grid = make_grid(output_image, nrow=2, padding=2, normalize=True)

# Option 1: Save directly with save_image
save_image(grid, "grid_image.png")

print()
