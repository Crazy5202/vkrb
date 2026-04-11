import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def local_magnitude_pruning(model, amount=0.3):
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)
                
    return model

def global_magnitude_pruning(model, amount=0.3):
    parameters_to_prune = []
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    return model

def random_pruning(model, amount=0.3):
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name='weight', amount=amount)
            
    return model

def make_pruning_permanent(model):
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias') and module.bias is not None and hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')

def check_sparsity(model):
    zeros = 0
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            zeros += torch.sum(module.weight == 0).item()
            total += module.weight.nelement()
    print(f"Разреженность модели: {100 * zeros / total:.2f}%")