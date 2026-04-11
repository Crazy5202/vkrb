from pytorch.imp.dataloader_test import test_dataloader
import pytorch.imp.metric as module_metric
import pytorch.imp.model as module_arch

# import pytorch.imp.loss as module_loss
# from pytorch.imp.trainer import Trainer
# from pytorch.utils.config import ConfigParser
# from pytorch.utils.util import prepare_device

import torch
import numpy as np
import time

def measure_inference_metric(model, data_loader, device, metric_fn, warmup_steps = False):
    """
    Measures average inference time per batch.
    """
    total_score = 0
    count = 0

    if warmup_steps:
        print("  Warming up...")
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                #if i >= warmup_steps: break
                data = data.to(device)
                _ = model(data)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()

    print("  Timing inference...")
    times = []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            #if i >= num_steps: break
            
            data = data.to(device)
            target = target.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            
            output = model(data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            score = metric_fn(output, target)
            total_score += score.item() if torch.is_tensor(score) else score
            count += 1

    avg_time_ms = (np.mean(times) * 1000)
    std_time_ms = (np.std(times) * 1000)
    return avg_time_ms, std_time_ms, total_score / count if count > 0 else 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2

MODEL_CONFIGS = [
    ("Original (Dense)", module_arch.HybridSN, "normal_best.pth"),
    ("TT CONV", module_arch.HybridSN_TT, "TT_best.pth"),
    ("Local Pruned", module_arch.HybridSN, "pruned_local.pth"),
    ("Global Pruned", module_arch.HybridSN, "pruned_global.pth"),
    ("Random Pruned", module_arch.HybridSN, "pruned_random.pth")
]

results = []

for name, model, path in MODEL_CONFIGS:
    print(f"Processing: {name}...")
    
    model = model(num_classes=NUM_CLASSES)
    try:
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    except FileNotFoundError:
        print(f"  File not found: {path}. Skipping.")
        continue

    model.to(DEVICE)
    model.eval()

    # train_dataloader = module_data.train_dataloader
    # valid_data_loader = train_dataloader.split_validation()

    metric_fn = module_metric.f1_score

    avg_time, std_time, metric_value = measure_inference_metric(model, test_dataloader, DEVICE, metric_fn)

    results.append({
        "Model": name,
        metric_fn.__str__: round(metric_value, 4),
        "Latency (ms)": round(avg_time, 3),
        "Std Dev (ms)": round(std_time, 3)
    })
print(results)