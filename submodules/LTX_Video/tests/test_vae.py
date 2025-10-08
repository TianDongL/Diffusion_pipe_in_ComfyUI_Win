import torch

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
    create_video_autoencoder_demo_config,
)


def test_vae():
    # create vae and run a forward and backward pass
    config = create_video_autoencoder_demo_config(latent_channels=16)
    video_autoencoder = CausalVideoAutoencoder.from_config(config)
    video_autoencoder.eval()
    input_videos = torch.randn(2, 3, 17, 64, 64)
    latent = video_autoencoder.encode(input_videos).latent_dist.mode()
    assert latent.shape == (2, 16, 3, 2, 2)
    timestep = torch.ones(input_videos.shape[0]) * 0.1
    reconstructed_videos = video_autoencoder.decode(
        latent, target_shape=input_videos.shape, timestep=timestep
    ).sample
    assert input_videos.shape == reconstructed_videos.shape
    loss = torch.nn.functional.mse_loss(input_videos, reconstructed_videos)
    loss.backward()

    # validate temporal causality in encoder
    input_image = input_videos[:, :, :1, :, :]
    image_latent = video_autoencoder.encode(input_image).latent_dist.mode()
    assert torch.allclose(image_latent, latent[:, :, :1, :, :], atol=1e-6)

    input_sequence = input_videos[:, :, :9, :, :]
    sequence_latent = video_autoencoder.encode(input_sequence).latent_dist.mode()
    assert torch.allclose(sequence_latent, latent[:, :, :2, :, :], atol=1e-6)
