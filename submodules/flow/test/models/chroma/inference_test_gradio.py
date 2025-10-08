import sys
import os
import gc
import warnings
import glob
import re
from datetime import datetime

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from safetensors.torch import safe_open
from transformers import T5Tokenizer
from torchvision.utils import save_image, make_grid
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import vae_flatten, prepare_latent_image_ids, vae_unflatten
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import gradio as gr

# Constants and paths
CHROMA_PATH = "models/flux/FLUX.1-schnell/chroma-unlocked-v6.safetensors"
VAE_PATH = "models/flux/ae.safetensors"
T5_PATH = "models/flux/text_encoder_2"
T5_CONFIG_PATH = "models/flux/text_encoder_2/config.json"
T5_TOKENIZER_PATH = "models/flux/tokenizer_2"
OUTPUT_DIR = "generated_images"
T5_MAX_LENGTH = 512
GUIDANCE_SCALE = 0  # Fixed guidance value

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_cuda_device():
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{torch.cuda.current_device()}"

def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Initialize models and tokenizer globally
t5_tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_PATH)
t5_config = T5Config.from_json_file(T5_CONFIG_PATH)
DEVICE = get_cuda_device()

# Initialize models in CPU memory
with torch.device("meta"):
    t5_model = T5EncoderModel(t5_config)
    chroma_model = Chroma(chroma_params)
    ae_model = AutoEncoder(ae_params)

# Load models to CPU
t5_model.load_state_dict(replace_keys(load_file_multipart(T5_PATH)), assign=True)
chroma_model.load_state_dict(load_safetensors(CHROMA_PATH), assign=True)
ae_model.load_state_dict(load_safetensors(VAE_PATH), assign=True)

# Keep track of last prompt and embeddings
last_prompt = None
last_embeddings = None

def generate_t5_embeddings(prompt, device=DEVICE):
    global last_prompt, last_embeddings, t5_model

    # Reuse embeddings if prompt hasn't changed
    if prompt == last_prompt and last_embeddings is not None:
        return last_embeddings.clone()

    t5_model.to(device)
    t5_model.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            text_inputs = t5_tokenizer(
                prompt,
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            t5_embed = t5_model(text_inputs.input_ids)
            t5_embed = t5_embed.clone().detach()

    t5_model.to('cpu')
    clear_cuda_memory()

    last_prompt = prompt
    last_embeddings = t5_embed.clone()

    return t5_embed

# Cache for generated images and their parameters
image_cache = {}

def get_cache_key(params):
    return f"{params['prompt']}_{params['seed']}_{params['width']}_{params['height']}_{params['steps']}_{params['guidance']}_{params['cfg']}_{params['first_n_steps_without_cfg']}"

def save_parameters(filename, params):
    param_file = filename.rsplit('.', 1)[0] + '.txt'
    with open(param_file, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def generate_random_seed():
    return torch.randint(0, 9999999, (1,)).item()

def update_seed_value(random_seed_enabled, current_seed):
    if random_seed_enabled:
        return torch.randint(0, 9999999, (1,)).item()
    return current_seed

def generate_image_with_params(prompt, seed, width, height, steps, cfg, first_n_steps_without_cfg, random_seed, progress=gr.Progress()):
    global chroma_model, ae_model

    parameters = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": GUIDANCE_SCALE,
        "cfg": cfg,
        "first_n_steps_without_cfg": first_n_steps_without_cfg,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    cache_key = get_cache_key(parameters)
    if cache_key in image_cache:
        return image_cache[cache_key]

    progress(0, desc="Initializing...")
    t5_embed = generate_t5_embeddings([prompt])

    progress(0.2, desc="Generating image...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chroma_model.to(DEVICE)
            ae_model.to(DEVICE)

            progress(0.4, desc="Processing...")

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            noise = get_noise(t5_embed.size(0), height, width, DEVICE, torch.bfloat16, seed)
            noise, shape = vae_flatten(noise)
            noise = noise.to(DEVICE)
            n, c, h, w = shape
            noise = noise.clone()

            image_pos_id = prepare_latent_image_ids(n, h, w).to(DEVICE)
            timesteps = get_schedule(steps, noise.shape[1])
            text_ids = torch.zeros((t5_embed.size(0), T5_MAX_LENGTH, 3), device=DEVICE)

            progress(0.6, desc="Denoising...")
            latent_cfg = denoise_cfg(
                chroma_model,
                noise,
                image_pos_id,
                t5_embed,
                torch.zeros_like(t5_embed),
                text_ids,
                timesteps,
                GUIDANCE_SCALE,
                cfg,
                first_n_steps_without_cfg,
            )

            progress(0.8, desc="Finalizing...")
            output_image = ae_model.decode(vae_unflatten(latent_cfg, shape))
            output_image = output_image.clone()

            grid = make_grid(output_image.clamp(-1, 1), nrow=2, padding=2, normalize=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_DIR}/generated_{timestamp}.png"
            save_image(grid, filename)
            save_parameters(filename, parameters)

            image_cache[cache_key] = filename

            chroma_model.to('cpu')
            ae_model.to('cpu')
            clear_cuda_memory()

            progress(1.0, desc="Done!")
            return filename

###########################
# Carousel Navigation     #
###########################

def carousel_navigate(direction, carousel_index, current_prompt, current_seed, current_width, current_height, current_steps, current_cfg, current_first_n_steps, current_image):
    """
    Build a carousel list that includes the current UI settings as the first item,
    then all saved images with their parameters from the OUTPUT_DIR.
    Cycle through them based on the given direction.
    """
    carousel_items = []
    carousel_items.append({
       "prompt": current_prompt,
       "seed": current_seed,
       "width": current_width,
       "height": current_height,
       "steps": current_steps,
       "cfg": current_cfg,
       "first_n_steps_without_cfg": current_first_n_steps,
       "random_seed": False,  # Disable random seed when loading a saved setting.
       "image": current_image
    })
    image_files = glob.glob(os.path.join(OUTPUT_DIR, "generated_*.png"))
    def extract_timestamp(fn):
        m = re.search(r"generated_(\d{8}_\d{6})\.png", os.path.basename(fn))
        return m.group(1) if m else ""
    image_files = sorted(image_files, key=extract_timestamp)
    for img_file in image_files:
        txt_file = img_file.rsplit('.', 1)[0] + ".txt"
        if os.path.exists(txt_file):
            params = {}
            with open(txt_file, "r") as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(":", 1)
                        params[key.strip()] = value.strip()
            try:
                params["seed"] = int(params.get("seed", current_seed))
            except:
                params["seed"] = current_seed
            try:
                params["width"] = int(params.get("width", current_width))
            except:
                params["width"] = current_width
            try:
                params["height"] = int(params.get("height", current_height))
            except:
                params["height"] = current_height
            try:
                params["steps"] = int(params.get("steps", current_steps))
            except:
                params["steps"] = current_steps
            try:
                params["cfg"] = float(params.get("cfg", current_cfg))
            except:
                params["cfg"] = current_cfg
            try:
                params["first_n_steps_without_cfg"] = int(params.get("first_n_steps_without_cfg", current_first_n_steps))
            except:
                params["first_n_steps_without_cfg"] = current_first_n_steps
            params["prompt"] = params.get("prompt", current_prompt)
            params["random_seed"] = False
            params["image"] = img_file
            carousel_items.append(params)
    new_index = (carousel_index + direction) % len(carousel_items)
    item = carousel_items[new_index]
    return (new_index,
            item["prompt"],
            item["seed"],
            item["width"],
            item["height"],
            item["steps"],
            item["cfg"],
            item["first_n_steps_without_cfg"],
            item["random_seed"],
            item["image"])

def carousel_left(carousel_index, current_prompt, current_seed, current_width, current_height, current_steps, current_cfg, current_first_n_steps, current_image):
    return carousel_navigate(-1, carousel_index, current_prompt, current_seed, current_width, current_height, current_steps, current_cfg, current_first_n_steps, current_image)

def carousel_right(carousel_index, current_prompt, current_seed, current_width, current_height, current_steps, current_cfg, current_first_n_steps, current_image):
    return carousel_navigate(1, carousel_index, current_prompt, current_seed, current_width, current_height, current_steps, current_cfg, current_first_n_steps, current_image)

###########################
# Gradio Interface        #
###########################

def create_interface():
    with gr.Blocks(title="Image Generation Interface") as interface:
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=(
                        "I'm looking for an artist to draw my anthro wolf character in a cyberpunk city at night. "
                        "He has dark grey fur with silver accents, yellow eyes, and a few scars. He wears a rugged "
                        "leather jacket with metallic shoulder pads, a tactical vest, cargo pants, and combat boots. "
                        "He's leaning against a brick wall in a rainy alley, smoking a cigarette with a tired but "
                        "confident expression. The neon lights cast a mix of blue and purple hues on him. Medium shot, "
                        "slightly low angle. Semi-realistic style with detailed textures"
                    ),
                    lines=3
                )
                with gr.Row():
                    with gr.Column():
                        random_seed = gr.Checkbox(label="Random Seed", value=False)
                        seed = gr.Slider(label="Seed", minimum=0, maximum=9999999, step=1, value=147302)
                        width = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=512)
                        height = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=512)
                    with gr.Column():
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=30)
                        cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=20, step=0.1, value=5.2)
                        first_n_steps_without_cfg = gr.Slider(label="First N Steps Without CFG", minimum=-1, maximum=50, step=1, value=6)
                generate_btn = gr.Button("Generate Image", variant="primary")
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="filepath")
                # Row for the arrow buttons (side by side)
                with gr.Row():
                    left_btn = gr.Button("←")
                    right_btn = gr.Button("→")
                # Hidden state to track carousel index
                carousel_index = gr.State(0)

        # Generation chain: update seed then generate image.
        generate_btn.click(
            fn=update_seed_value,
            inputs=[random_seed, seed],
            outputs=seed
        ).then(
            fn=generate_image_with_params,
            inputs=[prompt, seed, width, height, steps, cfg, first_n_steps_without_cfg, random_seed],
            outputs=output_image
        )

        # Carousel navigation for left/right buttons.
        left_btn.click(
            fn=carousel_left,
            inputs=[carousel_index, prompt, seed, width, height, steps, cfg, first_n_steps_without_cfg, output_image],
            outputs=[carousel_index, prompt, seed, width, height, steps, cfg, first_n_steps_without_cfg, random_seed, output_image]
        )
        right_btn.click(
            fn=carousel_right,
            inputs=[carousel_index, prompt, seed, width, height, steps, cfg, first_n_steps_without_cfg, output_image],
            outputs=[carousel_index, prompt, seed, width, height, steps, cfg, first_n_steps_without_cfg, random_seed, output_image]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
