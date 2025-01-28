import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

DEFAULT_LOSS_SCALE = 20.0

def convert_module_to_f16(module):
    """
    Recursively convert all layers in a module to float16 precision.
    """
    for child in module.children():
        convert_module_to_f16(child)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.half()

def convert_module_to_f32(module):
    """
    Recursively convert all layers in a module to float32 precision.
    """
    for child in module.children():
        convert_module_to_f32(child)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.float()

def set_module_to_fp16(module):
    """
    Convert layers to float16 precision.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()

def set_module_to_fp32(module):
    """
    Restore layers to float32 precision.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()

def create_master_params(param_shapes):
    """
    Generate a full-precision copy of model parameters.
    """
    master_params = []
    for group, shape in param_shapes:
        flat_param = _flatten_dense_tensors(
            [param.detach().float() for (_, param) in group]
        ).view(shape)
        param = nn.Parameter(flat_param, requires_grad=True)
        master_params.append(param)
    return master_params

def transfer_grads_to_master(param_shapes, master_params):
    """
    Copy gradients from model parameters to master parameters.
    """
    for master, (group, shape) in zip(master_params, param_shapes):
        master.grad = _flatten_dense_tensors(
            [safe_grad(param) for (_, param) in group]
        ).view(shape)

def sync_master_to_model(param_shapes, master_params):
    """
    Sync master parameter values to model parameters.
    """
    for master, (group, _) in zip(master_params, param_shapes):
        for (_, param), unflat_master in zip(
            group, unflatten_params(master, group)
        ):
            param.detach().copy_(unflat_master)

def unflatten_params(master_param, param_group):
    """
    Convert a flat tensor back to its original shape.
    """
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])

def parse_param_shapes(named_params):
    """
    Parse named model parameters into shape groups.
    """
    named_params = list(named_params)
    scalars = [(name, param) for name, param in named_params if param.ndim <= 1]
    matrices = [(name, param) for name, param in named_params if param.ndim > 1]
    return [(scalars, (-1)), (matrices, (1, -1))]

def initialize_master_from_state(model, state_dict, use_fp16):
    """
    Create master parameters from a saved state dictionary.
    """
    if use_fp16:
        named_params = [(name, state_dict[name]) for name, _ in model.named_parameters()]
        param_shapes = parse_param_shapes(named_params)
        return create_master_params(param_shapes)
    else:
        return [state_dict[name] for name, _ in model.named_parameters()]

def reset_master_grads(params):
    """
    Reset gradients for master parameters.
    """
    for param in params:
        param.grad = None

def reset_model_grads(model_params):
    """
    Reset gradients for model parameters.
    """
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def safe_grad(param):
    """
    Return gradients or zeros if gradients are None.
    """
    return param.grad.data.detach() if param.grad is not None else th.zeros_like(param)

class FP16Trainer:
    """
    Trainer class for mixed-precision training.
    """
    def __init__(self, model, use_fp16=False, scale_growth=1e-3, initial_loss_scale=DEFAULT_LOSS_SCALE):
        self.model = model
        self.use_fp16 = use_fp16
        self.scale_growth = scale_growth
        self.loss_scale = initial_loss_scale

        self.model_params = list(model.parameters())
        self.master_params = self.model_params
        self.param_shapes = None

        if use_fp16:
            self.param_shapes = parse_param_shapes(model.named_parameters())
            self.master_params = create_master_params(self.param_shapes)
            self.model.apply(set_module_to_fp16)

    def zero_gradients(self):
        """
        Reset gradients for model parameters.
        """
        reset_model_grads(self.model_params)

    def backward(self, loss):
        """
        Compute scaled gradients during backpropagation.
        """
        scaled_loss = loss * (2 ** self.loss_scale) if self.use_fp16 else loss
        scaled_loss.backward()

    def optimize(self, optimizer):
        """
        Optimize using either float16 or float32 parameters.
        """
        return self._fp16_optimize(optimizer) if self.use_fp16 else self._fp32_optimize(optimizer)

    def _fp16_optimize(self, optimizer):
        """
        Float16-specific optimization logic.
        """
        transfer_grads_to_master(self.param_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(2 ** self.loss_scale)

        if self._check_overflow(grad_norm):
            self.loss_scale -= 1
            reset_master_grads(self.master_params)
            return False

        self.master_params[0].grad.mul_(1.0 / (2 ** self.loss_scale))
        optimizer.step()
        reset_master_grads(self.master_params)
        sync_master_to_model(self.param_shapes, self.master_params)
        self.loss_scale += self.scale_growth
        return True

    def _fp32_optimize(self, optimizer):
        """
        Standard optimization for float32.
        """
        grad_norm, param_norm = self._compute_norms()
        optimizer.step()
        return True

    def _compute_norms(self, scale=1.0):
        """
        Compute gradient and parameter norms.
        """
        grad_norm = sum(
            th.norm(p.grad, p=2).item() ** 2 for p in self.master_params if p.grad is not None
        )
        param_norm = sum(
            th.norm(p, p=2).item() ** 2 for p in self.master_params
        )
        return np.sqrt(grad_norm) / scale, np.sqrt(param_norm)

    @staticmethod
    def _check_overflow(value):
        """
        Check for NaN or infinity in values.
        """
        return not th.isfinite(value).all()

    def save_state_dict(self):
        """
        Save master parameters to a state dictionary.
        """
        return {
            name: param.detach().cpu()
            for name, param in zip(self.model.state_dict().keys(), self.master_params)
        }

    def load_state_dict(self, state_dict):
        """
        Load state dictionary into master parameters.
        """
        self.master_params = initialize_master_from_state(self.model, state_dict, self.use_fp16)

    def reset_loss_scale(self):
        """
        Reset loss scale to its default value.
        """
        self.loss_scale = DEFAULT_LOSS_SCALE
