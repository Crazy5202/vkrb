from pytorch.imp.dataloader_test import test_dataloader
import pytorch.imp.metric as module_metric
import pytorch.imp.model as module_arch

import torch
import numpy as np
import time

#from thop import profile
#from fvcore.nn import FlopCountAnalysis

def measure_inference_metric(model, data_loader, device, metric_fns: list):
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

    #params = -1
    #flops = -1

    print("  Warming up...")
    with torch.no_grad():
        loader = data_loader

        data_iter = iter(loader)

        next_batch = next(data_iter)
        next_batch = [_.to(device, non_blocking=True) for _ in next_batch]

        #input = next_batch[0][0].unsqueeze(0)

        #flops = FlopCountAnalysis(model, input).total()

        #flops, params = profile(model, (input,), verbose=False)

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
                times.append((end_time - start_time)*1000)

            for i in range(len(metric_fns)):
                score = metric_fns[i](device, output, target)
                #print(score.item() if torch.is_tensor(score) else score)
                total_scores[i] += score.item() if torch.is_tensor(score) else score

            count += 1

    avg_time_ms = (np.mean(times))
    std_time_ms = (np.std(times))
    for i in range(len(metric_fns)):
        total_scores[i] = total_scores[i] / count if count > 0 else 0
    return avg_time_ms, std_time_ms, total_scores#, params, flops

def get_results(device, model_configs, metric_fns):

    results = []

    for name, model, path in model_configs:
        print(f"Processing: {name}...")

        try:
            checkpoint = torch.load(path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        except FileNotFoundError:
            print(f"  File not found: {path}. Skipping.")
            continue

        avg_time, std_time, metric_values = measure_inference_metric(model, test_dataloader, device, metric_fns)#, warmup_steps=False)

        res_dict = {
            "Model": name,
            "Latency (ms)": round(avg_time, 4),
            "Std Dev (ms)": round(std_time, 4),
            #"Million Parameters:": round(params/1e6, 4),
            #"GFLOPs": round(flops/1e9, 4)
        }

        for i in range(len(metric_fns)):
            res_dict[metric_fns[i].__name__] = round(metric_values[i], 4)

        results.append(res_dict)

    for entry in results:
        print(entry)

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_CONFIGS = [
        ("Original (Dense)", module_arch.SS_ConvNeXt_HSI_V2(), "results/models/best_ssv2.pth"),
        ("TT (With KD)", module_arch.get_convnext_tt(), "results/models/best_ssv2_tt_kd.pth"),
        ("Structured Pruning (With KD)", module_arch.get_convnext_pruned(), "results/models/best_ssv2_struct_pruned_kd.pth")
    ]

    METRIC_FNS = [module_metric.OA, module_metric.AA]

    get_results(DEVICE, MODEL_CONFIGS, METRIC_FNS)