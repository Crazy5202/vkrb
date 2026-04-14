import torch

from pytorch.src.base_trainer import BaseTrainer
from pytorch.utils.util import MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

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

        for batch_idx in range(len(loader)):

            (data, target) = next_batch 

            if batch_idx + 1 != len(loader): 

                next_batch = next(data_iter)
                next_batch = [ _.to(self.device, non_blocking=True) for _ in next_batch]

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.detach().item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(self.device, output, target))

            self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx),
                loss.detach().item()))

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

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.detach().item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self.device, output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        total = self.data_loader.n_samples
        current = min((batch_idx + 1) * self.data_loader.batch_size, total)
        return base.format(current, total, 100.0 * current / total)
