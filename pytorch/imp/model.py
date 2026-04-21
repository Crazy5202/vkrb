import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import tltorch

from pytorch.imp.convnext_ss_custom import SS_ConvNeXt_V2 as SS_ConvNeXt_HSI_V2, GRN

def replace_linear_with_tt_transformation(module, layer_name, rank):
    old_layer = getattr(module, layer_name)
    
    if not isinstance(old_layer, nn.Linear):
        return

    #print(f"Compressing Linear layer: {layer_name} (shape: {old_layer.weight.shape}) with TT-rank {rank}")

    try:
        tt_layer = tltorch.FactorizedLinear.from_linear(old_layer, rank=rank, factorization='blocktt', implementation='factorized')

        setattr(module, layer_name, tt_layer)
        
        #print(f"Success.")
        
    except Exception as e:
        print(f"Error during TT conversion: {e}")

def change_convnext_TT(model: nn.Module, rank_list=[1.0, 0.75, 0.5, 0.3]):

    rank_iter = iter(rank_list)

    for block_idx in range(len(model.main_layers)):

        r = next(rank_iter)
        
        for sub_idx in range(len(model.main_layers[block_idx])):
            
            try:
                replace_linear_with_tt_transformation(model.main_layers[block_idx][sub_idx], 'pwconv1', rank=r)
                replace_linear_with_tt_transformation(model.main_layers[block_idx][sub_idx], 'pwconv2', rank=r)
            except AttributeError:
                continue
            except IndexError:
                continue

def get_convnext_tt(rank_list=[1.0, 0.75, 0.5, 0.3]) -> nn.Module:
    SS_ConvNeXt_HSI_V2_TT = SS_ConvNeXt_HSI_V2()
    change_convnext_TT(SS_ConvNeXt_HSI_V2_TT, rank_list)
    return SS_ConvNeXt_HSI_V2_TT

# def prune_linear(module, layer_name, amount):
#     layer = getattr(module, layer_name)
    
#     if not isinstance(layer, nn.Linear):
#         return

#     #print(f"Pruning Linear layer: {layer_name} (shape: {layer.weight.shape}) with amount {amount}")

#     try:
#         prune.l1_unstructured(layer, 'weight', amount)
        
#         #print(f"Success.")
        
#     except Exception as e:
#         print(f"Error during pruning: {e}")

def prune_conv_block(block, amount=0.25):
    weight1 = block.pwconv1.weight.data
    importance = torch.norm(weight1, p=1, dim=1)
    
    old_hidden_dim = block.pwconv1.out_features
    new_hidden_dim = int(old_hidden_dim * (1 - amount))
    
    _, keep_idxs = torch.topk(importance, k=new_hidden_dim, sorted=True)
    keep_idxs = torch.sort(keep_idxs).values
    
    new_pw1 = nn.Linear(block.pwconv1.in_features, new_hidden_dim)
    new_pw1.weight.data = block.pwconv1.weight.data[keep_idxs]
    new_pw1.bias.data = block.pwconv1.bias.data[keep_idxs]
    
    new_grn = GRN(new_hidden_dim)
    new_grn.gamma.data = block.grn.gamma.data[:, :, :, keep_idxs]
    new_grn.beta.data = block.grn.beta.data[:, :, :, keep_idxs]
    
    new_pw2 = nn.Linear(new_hidden_dim, block.pwconv2.out_features)
    new_pw2.weight.data = block.pwconv2.weight.data[:, keep_idxs]
    new_pw2.bias.data = block.pwconv2.bias.data
    
    block.pwconv1 = new_pw1
    block.grn = new_grn
    block.pwconv2 = new_pw2
    
    return block

def prune_convnext(model: nn.Module, rank_list=[1.0, 0.75, 0.5, 0.3]):

    rank_iter = iter(rank_list)

    for block_idx in range(len(model.main_layers)):

        r = next(rank_iter)
        
        for sub_idx in range(len(model.main_layers[block_idx])):
            
            try:
                prune_conv_block(model.main_layers[block_idx][sub_idx], 1.0-r)
            except Exception as e:
                print(f"Ошибка при структурированном прунинге блока {block_idx}.{sub_idx}: {e}")

def get_convnext_pruned(rank_list=[1.0, 0.75, 0.5, 0.3]) -> nn.Module:
    SS_ConvNeXt_HSI_V2_pruned = SS_ConvNeXt_HSI_V2()
    prune_convnext(SS_ConvNeXt_HSI_V2_pruned, rank_list)
    return SS_ConvNeXt_HSI_V2_pruned

# def apply_pruning_sparse(model: nn.Module):
#     for _, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             if hasattr(module, 'weight_mask'):
#                 prune.remove(module, 'weight')
#                 module.weight = module.weight.to_sparse()
#             if hasattr(module, 'bias') and module.bias is not None and hasattr(module, 'bias_mask'):
#                 prune.remove(module, 'bias')