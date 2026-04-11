import torch

from pytorch.imp.model import HybridSN
from pytorch.utils.prune import *

CHECKPOINT_PATH = "normal_best.pth"
PRUNING_AMOUNT = 0.3

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

base_state_dict = checkpoint['state_dict']

experiments = [
    ("Local_Magnitude", local_magnitude_pruning, "pruned_local.pth"),
    ("Global_Magnitude", global_magnitude_pruning, "pruned_global.pth"),
    ("Random", random_pruning, "pruned_random.pth")
]

for name, prune_fn, save_path in experiments:
    print(f"\n{name}")
    
    model = HybridSN(num_classes=2)
    model.load_state_dict(base_state_dict)
    
    pruned_model = prune_fn(model, amount=PRUNING_AMOUNT)
    
    make_pruning_permanent(pruned_model)
    
    check_sparsity(pruned_model)
    
    torch.save({'state_dict': pruned_model.state_dict()}, save_path)