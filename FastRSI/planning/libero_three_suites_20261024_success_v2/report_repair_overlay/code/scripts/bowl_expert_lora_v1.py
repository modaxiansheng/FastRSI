"""Rank-four Q/V adapters for the frozen native action expert.

Independent implementation of LoRA, not official OpenPI PyTorch LoRA support.
Base matrices keep their native dtype; residual parameters and computation use
FP32, then the residual is cast to the native layer output dtype.
"""
import math
import torch
import torch.nn.functional as F

PREFIX = 'paligemma_with_expert.gemma_expert.'
TARGETS = ('q_proj', 'v_proj')
RANK = 4

class ExpertLoRALinear(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.enabled = True
        self.lora_A = torch.nn.Parameter(torch.empty(RANK, base.in_features,
            device=base.weight.device, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, RANK,
            device=base.weight.device, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
            y = y + delta.to(y.dtype)
        return y

def install(model):
    for p in model.parameters():
        p.requires_grad_(False)
    modules = {}
    for name, module in list(model.named_modules()):
        if name.startswith(PREFIX) and name.split('.')[-1] in TARGETS:
            assert isinstance(module, torch.nn.Linear), (name, type(module))
            parent_name, attribute = name.rsplit('.', 1)
            replacement = ExpertLoRALinear(module)
            setattr(model.get_submodule(parent_name), attribute, replacement)
            modules[name] = replacement
    assert len(modules) == 36, len(modules)
    return modules

def enable(modules, active):
    for module in modules.values():
        module.enabled = active

def adapter_tensors(modules):
    return {name+'.'+key: getattr(module, key)
            for name, module in modules.items() for key in ('lora_A','lora_B')}

def load_adapter(model, path):
    from safetensors.torch import load_file
    modules = install(model)
    params = adapter_tensors(modules)
    weights = load_file(str(path), device='cpu')
    assert set(weights) == set(params)
    with torch.no_grad():
        for name, parameter in params.items():
            source = weights[name]
            assert source.shape == parameter.shape and source.dtype == torch.float32
            assert torch.isfinite(source).all()
            parameter.copy_(source.to(parameter.device))
    model.eval()
    return modules
