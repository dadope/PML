import torch
from guided_diffusion.conditioning_registry import BaseConditioning, register_conditioning
from utils.img_utils import normalize, dynamic_thresholding

def dynamic_thresholding(x, threshold=1.0):
    """
    Apply dynamic thresholding to clip extreme values.
    """
    x = torch.clamp(x, -threshold, threshold)
    return x / max(x.abs().max(), threshold)

@register_conditioning("identity")
class IdentityConditioning(BaseConditioning):
    """
    Identity method: passes the input unaltered.
    """
    def condition(self, x_t, **kwargs):
        return x_t.to(x_t.device)  # Ensure x_t stays on the correct device

@register_conditioning("projection")
class ProjectionConditioning(BaseConditioning):
    """
    Projection-based conditioning.
    """
    def condition(self, x_t, noisy_measurement, **kwargs):
        x_t = x_t.to(noisy_measurement.device)  # Match device with noisy_measurement
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
        x_prev = x_prev.to(measurement.device)
        x_t = x_t.to(measurement.device)
        x_0_hat = x_0_hat.to(measurement.device)

        gradient, norm = self.compute_gradient_and_norm(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
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
        """
        Apply posterior sampling conditioning.
        """
        # Ensure tensors are on the same device
        x_prev = x_prev.to(measurement.device).requires_grad_(True)
        x_t = x_t.to(measurement.device)
        x_0_hat = x_0_hat.to(measurement.device)

        # Compute gradient and norm
        gradient, norm = self.compute_gradient_and_norm(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )

        # Apply dynamic thresholding to stabilize intermediate results
        x_0_hat = dynamic_thresholding(x_0_hat)

        # Update x_t using the gradient
        x_t = x_t - self.scale * gradient.detach()

        # Normalize to stabilize values
        x_t = normalize(x_t)

        # Return updated x_t and norm for logging
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
        # Ensure tensors are on the same device
        x_prev = x_prev.to(measurement.device).requires_grad_(True)
        x_t = x_t.to(measurement.device)
        x_0_hat = x_0_hat.to(measurement.device)

        # Compute gradient across samples
        total_norm = 0
        total_gradient = torch.zeros_like(x_prev, device=x_prev.device)
        for _ in range(self.num_samples):
            noisy_x0 = x_0_hat + 0.05 * torch.randn_like(x_0_hat, device=x_0_hat.device)
            diff = measurement - self.operator.forward(noisy_x0)
            norm = torch.linalg.norm(diff) / self.num_samples
            total_norm += norm.item()
            total_gradient += torch.autograd.grad(
                outputs=norm, inputs=x_prev, retain_graph=True, allow_unused=True
            )[0]

        x_t = x_t - self.scale * total_gradient.detach()  # Detach gradient to free memory
        return x_t.detach(), total_norm

