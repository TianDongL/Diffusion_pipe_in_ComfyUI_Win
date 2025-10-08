import pytest
import torch
from ltx_video.schedulers.rf import RectifiedFlowScheduler


def init_latents_and_scheduler(sampler):
    batch_size, n_tokens, n_channels = 2, 4096, 128
    num_steps = 20
    scheduler = RectifiedFlowScheduler(
        sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
    )
    latents = torch.randn(size=(batch_size, n_tokens, n_channels))
    scheduler.set_timesteps(num_inference_steps=num_steps, samples_shape=latents.shape)
    return scheduler, latents


@pytest.mark.parametrize("sampler", ["LinearQuadratic", "Uniform"])
def test_scheduler_default_behavior(sampler):
    """
    Test the case of a single timestep from the list of timesteps.
    """
    scheduler, latents = init_latents_and_scheduler(sampler)

    for i, t in enumerate(scheduler.timesteps):
        noise_pred = torch.randn_like(latents)
        denoised_latents = scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
        )[0]

        # Verify the denoising
        next_t = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else 0.0
        dt = t - next_t
        expected_denoised_latents = latents - dt * noise_pred
        assert torch.allclose(denoised_latents, expected_denoised_latents, atol=1e-06)


@pytest.mark.parametrize("sampler", ["LinearQuadratic", "Uniform"])
def test_scheduler_per_token(sampler):
    """
    Test the case of a timestep per token (from the list of timesteps).
    Some tokens are set with timestep of 0.
    """
    scheduler, latents = init_latents_and_scheduler(sampler)
    batch_size, n_tokens = latents.shape[:2]
    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.full((batch_size, n_tokens), t)
        timesteps[:, 0] = 0.0
        noise_pred = torch.randn_like(latents)
        denoised_latents = scheduler.step(
            noise_pred,
            timesteps,
            latents,
            return_dict=False,
        )[0]

        # Verify the denoising
        next_t = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else 0.0
        next_timesteps = torch.full((batch_size, n_tokens), next_t)
        dt = timesteps - next_timesteps
        expected_denoised_latents = latents - dt.unsqueeze(-1) * noise_pred
        assert torch.allclose(
            denoised_latents[:, 1:], expected_denoised_latents[:, 1:], atol=1e-06
        )
        assert torch.allclose(denoised_latents[:, 0], latents[:, 0], atol=1e-06)


@pytest.mark.parametrize("sampler", ["LinearQuadratic", "Uniform"])
def test_scheduler_t_not_in_list(sampler):
    """
    Test the case of a timestep per token NOT from the list of timesteps.
    """
    scheduler, latents = init_latents_and_scheduler(sampler)
    batch_size, n_tokens = latents.shape[:2]
    for i in range(len(scheduler.timesteps)):
        if i < len(scheduler.timesteps) - 1:
            t = (scheduler.timesteps[i] + scheduler.timesteps[i + 1]) / 2
        else:
            t = scheduler.timesteps[i] / 2
        timesteps = torch.full((batch_size, n_tokens), t)
        noise_pred = torch.randn_like(latents)
        denoised_latents = scheduler.step(
            noise_pred,
            timesteps,
            latents,
            return_dict=False,
        )[0]

        # Verify the denoising
        next_t = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else 0.0
        next_timesteps = torch.full((batch_size, n_tokens), next_t)
        dt = timesteps - next_timesteps
        expected_denoised_latents = latents - dt.unsqueeze(-1) * noise_pred
        assert torch.allclose(denoised_latents, expected_denoised_latents, atol=1e-06)
