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

def measure_inference_metric(model, data_loader, device, metric_fns: list, warmup_steps = True):
    """
    Measures average inference time per batch.
    """
    total_scores = [0]*len(metric_fns)
    count = 0

    model.to(device)
    model.eval()

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    if warmup_steps:
        print("  Warming up...")
        with torch.no_grad():
            loader = data_loader

            data_iter = iter(loader)

            next_batch = next(data_iter)
            next_batch = [_.to(device, non_blocking=True) for _ in next_batch]

            for batch_idx in range(len(loader)):

                (data, target) = next_batch 

                if batch_idx + 1 != len(loader): 

                    next_batch = next(data_iter)
                    next_batch = [ _.to(device, non_blocking=True) for _ in next_batch]

                output = model(data)

    print("  Timing inference...")
    times = []
    with torch.no_grad():
        loader = data_loader

        data_iter = iter(loader)

        next_batch = next(data_iter)
        next_batch = [_.to(device, non_blocking=True) for _ in next_batch]

        for batch_idx in range(len(loader)):

            (data, target) = next_batch 

            if batch_idx + 1 != len(loader): 

                next_batch = next(data_iter)
                next_batch = [ _.to(device, non_blocking=True) for _ in next_batch]

            if device.type == 'cuda':
                start_event.record()
            else:
                start_time = time.perf_counter()

            output = model(data)

            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            else:
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            for i in range(len(metric_fns)):
                score = metric_fns[i](device, output, target)
                total_scores[i] += score.item() if torch.is_tensor(score) else score

            count += 1

    avg_time_ms = (np.mean(times))
    std_time_ms = (np.std(times))
    for i in range(len(metric_fns)):
        total_scores[i] = total_scores[i] / count if count > 0 else 0
    return avg_time_ms, std_time_ms, total_scores

def get_results(device, model_configs, metric_fns):

    results = []

    for name, model, path in model_configs:
        print(f"Processing: {name}...")
        
        model = model()

        try:
            checkpoint = torch.load(path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        except FileNotFoundError:
            print(f"  File not found: {path}. Skipping.")
            continue

        # train_dataloader = module_data.train_dataloader
        # valid_data_loader = train_dataloader.split_validation()

        avg_time, std_time, metric_values = measure_inference_metric(model, test_dataloader, device, metric_fns)

        res_dict = {
            "Model": name,
            "Latency (ms)": round(avg_time, 4),
            "Std Dev (ms)": round(std_time, 4)
        }

        for i in range(len(metric_fns)):
            res_dict[metric_fns[i].__name__] = round(metric_values[i], 4)

        results.append(res_dict)

    print(results)

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_CONFIGS = [
        ("Original (Dense)", module_arch.SS_ConvNeXt_HSI, "ss_2242_32_best.pth"),
    ]

    METRIC_FNS = [module_metric.OA, module_metric.AA]

    get_results(DEVICE, MODEL_CONFIGS, METRIC_FNS)