import torch
from guided_diffusion.conditioning_registry import BaseConditioning, register_conditioning
from utils.img_utils import normalize, dynamic_thresholding

@register_conditioning("identity")
class IdentityConditioning(BaseConditioning):
    """
    Identity method: passes the input unaltered.
    """
    def condition(self, x_t, **kwargs):
        return x_t

@register_conditioning("projection")
class ProjectionConditioning(BaseConditioning):
    """
    Projection-based conditioning.
    """
    def condition(self, x_t, noisy_measurement, **kwargs):
        return self.apply_projection(data=x_t, noisy_measurement=noisy_measurement, **kwargs)

@register_conditioning("mcg")
class ManifoldConstraintGradient(BaseConditioning):
    """
    Gradient-based conditioning on the manifold constraint.
    """
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.scale = kwargs.get("scale", 1.0)

    def condition(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        gradient, norm = self.compute_gradient_and_norm(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= self.scale * gradient
        x_t = self.apply_projection(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm

@register_conditioning("ps")
class PosteriorSampling(BaseConditioning):
    """
    Posterior sampling method.
    """
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.scale = kwargs.get("scale", 1.0)

    def condition(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        gradient, norm = self.compute_gradient_and_norm(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= self.scale * gradient
        return x_t, norm

@register_conditioning("ps+")
class PosteriorSamplingPlus(BaseConditioning):
    """
    Enhanced posterior sampling with gradient noise.
    """
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.num_samples = kwargs.get("num_samples", 5)
        self.scale = kwargs.get("scale", 1.0)

    def condition(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        total_norm = 0
        for _ in range(self.num_samples):
            noisy_x0 = x_0_hat + 0.05 * torch.randn_like(x_0_hat)
            diff = measurement - self.operator.forward(noisy_x0)
            total_norm += torch.linalg.norm(diff) / self.num_samples
        gradient = torch.autograd.grad(outputs=total_norm, inputs=x_prev, retain_graph=True)[0]
        x_t -= self.scale * gradient
        return x_t, total_norm
