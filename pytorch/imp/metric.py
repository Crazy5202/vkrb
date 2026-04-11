import torch
#from torchmetrics import F1Score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        TP = ((pred == 1) & (target == 1)).sum().item()
        FP = ((pred == 1) & (target == 0)).sum().item()
        FN = ((pred == 0) & (target == 1)).sum().item()

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# def f1(output, target):
#     f1_metric = F1Score(task='binary')
#     with torch.no_grad():
#         output_classes = torch.argmax(output, dim=1)
#         f1_value = f1_metric(output_classes, target)
#     return f1_value

# def average_accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         acc_per_class = []
        
#         for c in range(2):
#             mask = (target == c)
#             if mask.sum() > 0:
#                 class_acc = (pred[mask] == c).float().mean()
#                 acc_per_class.append(class_acc)
        
#         if len(acc_per_class) == 0:
#             return 0.0
            
#         return torch.stack(acc_per_class).mean().item()

# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

# def f1(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
#     '''
#     Calculate F1 score. Can work with gpu tensors
#     Returns
#     -------
#     torch.Tensor
#         `ndim` == 1. 0 <= val <= 1
    
#     '''
#     if y_pred.ndim == 2:
#         y_pred = y_pred.argmax(dim=1)
    
#     tp = (y_true * y_pred).sum().to(torch.float32)
#     fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
#     fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
#     epsilon = 1e-9
    
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)
    
#     f1 = 2* (precision*recall) / (precision + recall + epsilon)
#     return f1