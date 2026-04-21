import torch
import torch.nn.functional as F

from pytorch.src.base_trainer import BaseTrainer
from pytorch.utils.util import MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, bf16: bool = False,
                 grad_clip_val = None, grad_accum_steps=1, 
                 kd_model: torch.nn.Module = None, kd_temperature=2.0, kd_alpha = 0.7):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler

        self.bf16 = bf16

        self.grad_clip_val = grad_clip_val
        self.grad_accum_steps = grad_accum_steps

        self.teacher = kd_model
        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _distillation_loss(self, student_logits, teacher_logits, targets):
        soft_t = F.softmax(teacher_logits / self.kd_temperature, dim=-1)

        log_s = F.log_softmax(student_logits / self.kd_temperature, dim=-1)

        soft_loss = F.kl_div(log_s, soft_t, reduction="batchmean") * (self.kd_temperature ** 2)

        hard_loss = self.criterion(student_logits, targets)

        loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss
        return loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        loader = self.data_loader

        data_iter = iter(loader)

        next_batch = next(data_iter)
        next_batch = [_.to(self.device, non_blocking=True) for _ in next_batch]

        self.optimizer.zero_grad()

        for batch_idx in range(len(loader)):

            (data, target) = next_batch 

            if batch_idx + 1 != len(loader): 

                next_batch = next(data_iter)
                next_batch = [ _.to(self.device, non_blocking=True) for _ in next_batch]
            
            if self.teacher is None:

                if self.bf16 == True:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss = loss / self.grad_accum_steps
                        loss.backward()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.grad_accum_steps
                    loss.backward()

            else:

                with torch.no_grad():
                    if self.bf16:
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            teacher_logits = self.teacher(data)
                    else:
                        teacher_logits = self.teacher(data)

                if self.bf16:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        student_logits = self.model(data)
                        loss = self._distillation_loss(student_logits, teacher_logits, target)
                        loss = loss / self.grad_accum_steps
                        loss.backward()
                else:
                    student_logits = self.model(data)
                    loss = self._distillation_loss(student_logits, teacher_logits, target)
                    loss = loss / self.grad_accum_steps
                    loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.grad_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_val)

                self.optimizer.step()
                self.optimizer.zero_grad()

            unscaled_loss = loss.detach().item() * self.grad_accum_steps

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', unscaled_loss)
            for met in self.metric_ftns:
                out_for_metrics = student_logits if self.teacher is not None else output
                met_value = met(self.device, out_for_metrics, target)
                self.train_metrics.update(met.__name__, met_value)

            self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx),
                unscaled_loss))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():

            loader = self.valid_data_loader

            data_iter = iter(loader)

            next_batch = next(data_iter)
            next_batch = [_.to(self.device, non_blocking=True) for _ in next_batch]

            for batch_idx in range(len(loader)):

                (data, target) = next_batch 

                if batch_idx + 1 != len(loader): 

                    next_batch = next(data_iter)
                    next_batch = [ _.to(self.device, non_blocking=True) for _ in next_batch]

                if self.bf16 == True:
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.detach().item())
                for met in self.metric_ftns:
                    met_value = met(self.device, output, target)
                    self.valid_metrics.update(met.__name__, met_value)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        total = self.data_loader.n_samples
        current = min((batch_idx + 1) * self.data_loader.batch_size, total)
        return base.format(current, total, 100.0 * current / total)
