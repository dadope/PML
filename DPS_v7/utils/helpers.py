import math
import numpy as np
import torch

def extract_and_expand(array, t, target):
    """
    Extract values from a 1-D tensor for specific timesteps and expand them to match the shape of the target tensor.

    :param array: A 1-D PyTorch tensor.
    :param t: A tensor of indices corresponding to the timesteps.
    :param target: A tensor whose shape will be matched.
    :return: A tensor with the same shape as the target, containing the extracted values.
    """
    # Ensure `array` is on the same device as `target`
    array = array.to(target.device)[t].float()

    # Expand dimensions to match `target`'s shape
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target)

def space_timesteps(num_timesteps, section_counts):
    """
    Generate a set of timesteps from a diffusion process, optionally spaced differently.

    :param num_timesteps: Total number of timesteps in the diffusion process.
    :param section_counts: Specifies the spacing strategy. Can be:
                           - A string starting with "ddim" (e.g., "ddim50") for fixed striding.
                           - An integer for evenly spaced sections.
    :return: A set of selected timesteps.
    """
    if isinstance(section_counts, str) and section_counts.startswith("ddim"):
        desired_count = int(section_counts[4:])
        stride = num_timesteps // desired_count
        return set(range(0, num_timesteps, stride))
    elif isinstance(section_counts, int):
        stride = num_timesteps // section_counts
        return set(range(0, num_timesteps, stride))
    else:
        raise ValueError(f"Unsupported section_counts value: {section_counts}")

def get_named_beta_schedule(schedule_name, num_timesteps):
    """
    Get a predefined beta schedule for diffusion processes.

    :param schedule_name: The name of the schedule (e.g., "linear", "cosine").
    :param num_timesteps: The number of timesteps for the schedule.
    :return: A numpy array of beta values.
    """
    if schedule_name == "linear":
        beta_start, beta_end = 0.0001, 0.02
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        alphas = [alpha_bar_fn(t / num_timesteps) for t in range(num_timesteps)]
        betas = []
        for i in range(1, num_timesteps):
            betas.append(min(1 - alphas[i] / alphas[i - 1], 0.999))
        return np.array(betas, dtype=np.float64)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")
