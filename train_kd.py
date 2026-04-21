from pytorch.imp.dataloader_train import train_dataloader
import pytorch.imp.loss as module_loss
import pytorch.imp.metric as module_metric
import pytorch.imp.model as module_arch
from pytorch.imp.trainer import Trainer
from pytorch.utils.config import ConfigParser
from pytorch.utils.util import prepare_device, model_info

import torch
import numpy as np

SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# "use_kd": true,
# "kd_temperature": 2.0,
# "kd_alpha": 0.7,

def main(config: ConfigParser):
    logger = config.get_logger('train')

    valid_data_loader = train_dataloader.split_validation()

    # model = module_arch.get_convnext_tt()
    # checkpoint = torch.load("best_ssv2_tt.pth", weights_only=False)
    # model.load_state_dict(checkpoint['state_dict'])

    model = module_arch.get_convnext_pruned()
    checkpoint = torch.load("results/models/best_ssv2_struct_pruned.pth", weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    logger.info(model_info(model))

    model_teacher = module_arch.SS_ConvNeXt_HSI_V2()
    checkpoint = torch.load("results/models/best_ssv2.pth", weights_only=False)
    model_teacher.load_state_dict(checkpoint['state_dict'])

    device, device_ids = prepare_device(config['n_gpu'])
    
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model_teacher = model_teacher.to(device)
    if len(device_ids) > 1:
        model_teacher = torch.nn.DataParallel(model_teacher, device_ids=device_ids)

    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    if config['warmup'] == True:
        warmup_scheduler = config.init_obj('warmup_scheduler', torch.optim.lr_scheduler, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[config['warmup_scheduler']['args']['total_iters']]
        )

    bf16_flag = config['bf16']

    grad_clip_val = None
    if config['grad_clip'] == True:
        grad_clip_val = config['grad_clip']

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler, bf16=bf16_flag, 
                      grad_accum_steps=config['grad_accum_steps'],
                      grad_clip_val=grad_clip_val,
                      kd_model=model_teacher)

    trainer.train()

if __name__ == '__main__':
    
    model_path = "."#saved\models\Squeeze_MobileNet_V3\\0417_152013"
    config_path = model_path + "\\config.json"
    resume_path = None#model_path + "\\model_best.pth" #"\\checkpoint-epoch5.pth"
    config = ConfigParser(config_path, resume=resume_path)

    main(config)