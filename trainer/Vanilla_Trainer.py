import numpy as np
import torch

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Vanilla_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self):
        pass

    def set_param(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.val_all, self.val_low, self.val_high = valid_data_loader
        self.do_validation = self.val_all is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_all_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_low_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_high_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):
            # start, end = data[:, :140, :4], data[:, 140:280]
            seqs = data[:, :280].view(-1, 2, 140, 4)
            lens, target = data[:, 280, :3], data[:, 280, 3]

            seqs, lens, target = seqs.to(self.device), lens.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(seqs, lens)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # logs that this training step has been taken at current time
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                try:
                    auc_val = met(output, target)
                    self.train_metrics.update(met.__name__, auc_val)
                except ValueError:
                    print('AUC bitching around for train metrics')
                    continue

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # will make problems for non-image data
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log_all, val_log_low, val_log_high = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log_all.items()})
            val_log_low.pop('loss', None)
            log.update(**{'val_low_' + k: v for k, v in val_log_low.items()})
            val_log_high.pop('loss', None)
            log.update(**{'val_high_' + k: v for k, v in val_log_high.items()})

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
        self.valid_all_metrics.reset()
        self.valid_low_metrics.reset()
        self.valid_high_metrics.reset()
        with torch.no_grad():
            self._single_val_epoch(self.val_all, epoch, self.valid_all_metrics)
            self._single_val_epoch(self.val_low, epoch, self.valid_low_metrics)
            self._single_val_epoch(self.val_high, epoch, self.valid_high_metrics)


        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_all_metrics.result(), self.valid_low_metrics.result(), self.valid_high_metrics.result()

    def _single_val_epoch(self, val_data, epoch, metrics):
        for batch_idx, data in enumerate(val_data):
            seqs = data[:, :280].view(-1, 2, 140, 4)
            lens, target = data[:, 280, :3], data[:, 280, 3]

            seqs, lens, target = seqs.to(self.device), lens.to(self.device), target.to(self.device)

            output = self.model(seqs, lens)
            loss = self.criterion(output, target)

            self.writer.set_step((epoch - 1) * len(val_data) + batch_idx, 'valid')
            metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                try:
                    auc_val = met(output, target)
                    metrics.update(met.__name__, auc_val)
                except ValueError:
                    print('AUC bitching around')
                    continue

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
