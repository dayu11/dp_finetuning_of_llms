
import torch
import torch.nn as nn

from .lora_utils import lora_linear

def linear_forward_hook(module, intsr, outtsr):
    module.input = intsr[0]

def linear_backward_hook(layer, grad_input, grad_output):
    grad_output = grad_output[0] # n, len, outdim
    grad_input = layer.input #n, len, indim

    layer_batch_dim = 0

    A = grad_input
    B = grad_output

    # Compute per-sequence gradients
    # The gradients of tokens in the same sequence are summed up
    # k: tokens-per-sample
    # n: batch size
    if layer_batch_dim == 1:
        gs = torch.einsum("kn...i,kn...j->nij", B, A)
        if layer.bias is not None:
            gs_bias = torch.einsum("kn...i->ni", B)
    else:
        gs = torch.einsum("n...i,n...j->nij", B, A)
        if layer.bias is not None:
            gs_bias = torch.einsum("n...k->nk", B)

    layer.weight.grad_sample = gs.float()
    if layer.bias is not None:
        layer.bias.grad_sample = gs_bias.float()

def make_lora_model_dp(model):
    # register forward and backward hooks for lora branch
    for module in model.modules():
        if isinstance(module, lora_linear):
            module.lora_branch_in.register_forward_hook(linear_forward_hook)
            module.lora_branch_in.register_backward_hook(linear_backward_hook)
            module.lora_branch_out.register_forward_hook(linear_forward_hook)
            module.lora_branch_out.register_backward_hook(linear_backward_hook)

def get_grad_norm(params):
    for p in params:
        if(hasattr(p, 'grad_sample')):
            n = p.grad_sample.shape[0]
            break
    grad_norm_list = torch.zeros(n).cuda()
    for p in params: 
        if(hasattr(p, 'grad_sample')):
            flat_g = p.grad_sample.reshape(n, -1)
            current_norm_list = torch.norm(flat_g, dim=1)
            grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)

    return grad_norm_list

def clip_grad_sample(params, clipping):
    for p in params:
        if(hasattr(p, 'grad_sample')):
            n = p.grad_sample.shape[0]
            break
    grad_norm_list = torch.zeros(n).cuda()
    for p in params: 
        if(hasattr(p, 'grad_sample')):
            flat_g = p.grad_sample.reshape(n, -1)
            current_norm_list = torch.norm(flat_g, dim=1)
            grad_norm_list += torch.square(current_norm_list)
            #print(current_norm_list[0].item(), p.shape, p.numel())
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        if(hasattr(p, 'grad_sample')):
            p_dim = len(p.shape)
            scaling = scaling.view([n] + [1]*p_dim)
            p.grad_sample *= scaling

    return grad_norm_list

def get_epsilon_prv(noise_multiplier, delta, steps, sampling_prob):
    from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
    from prv_accountant import PRVAccountant

    prv = PoissonSubsampledGaussianMechanism(noise_multiplier=noise_multiplier, sampling_probability=sampling_prob)
    accountant = PRVAccountant(
        prvs=[prv],
        max_self_compositions=[steps],
        eps_error=0.1,
        delta_error=delta/10
    )    
    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=delta, num_self_compositions=[steps]) 
    return eps_up


def clip_and_accumulate_perexample_grads(require_grad_params, accumulated_steps, clip_norm, accelerator):

    if accelerator.scaler is not None:
        # get the scale of mixed precision training
        mixed_precision_scale = accelerator.scaler.get_scale()
    else:
        mixed_precision_scale = 1.0
    for p in require_grad_params:
        if hasattr(p, 'grad_sample'):
            # convert to fp32
            p.grad_sample = p.grad_sample.float()
            # undo mixed precision scaling
            p.grad_sample /= mixed_precision_scale
            # undo loss averaging
            p.grad_sample *= p.grad_sample.shape[0]
        else:
            raise RuntimeError("DP enabled but no grad_sample found")
    # clip gradients
    grad_norms = clip_grad_sample(require_grad_params, clip_norm)
    # accumulate gradients
    for p in require_grad_params:
        if hasattr(p, 'grad_sample'):
            if accumulated_steps == 0:
                p.accumulated_grad = torch.sum(p.grad_sample, dim=0)
            else:
                p.accumulated_grad += torch.sum(p.grad_sample, dim=0)
            p.grad_sample = None
        else:
            raise RuntimeError("DP enabled but no grad_sample found")
        
    return grad_norms

def _add_noise(tsr, clip_norm, noise_multiplier):
    # add noise
    noise = torch.normal(0, clip_norm * noise_multiplier, size=tsr.shape, device=tsr.device, dtype=tsr.dtype)
    tsr = tsr + noise
    return tsr

def add_noise_to_grads(grads, clip_norm, noise_multiplier):
    noisy_grads = [_add_noise(g, clip_norm, noise_multiplier) for g in grads]
    return noisy_grads

