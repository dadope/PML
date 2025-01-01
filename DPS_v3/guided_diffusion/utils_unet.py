import torch as th


def cal_gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10.0, mode="mixed"):
    """
    Calculate gradient penalty for WGAN-GP.
    """
    if lambda_gp <= 0:
        return 0.0, None

    interpolates = real_data if mode == "real" else fake_data
    if mode == "mixed":
        alpha = th.rand(real_data.shape[0], 1, device=device).expand_as(real_data)
        interpolates = alpha * real_data + (1 - alpha) * fake_data

    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = th.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=th.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(real_data.size(0), -1)
    penalty = (((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp)
    return penalty, gradients
