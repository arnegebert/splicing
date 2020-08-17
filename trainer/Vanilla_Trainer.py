import numpy as np
import torch

from base import BaseTrainer
from utils import inf_loop, MetricTracker, save_pred_and_target
from visualizations.roc_curves import plot_and_save_roc


class Vanilla_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self):
        pass

    def set_param(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, four_seq=False, embedded=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.embedded = embedded
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.test_all, self.test_low, self.test_high, self.val_all = valid_data_loader
        self.do_validation = self.val_all is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.four_seq = four_seq

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_all_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_low_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_high_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)

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
            if self.embedded:
                seqs = data[:, :2].view(-1, 200)
                lens, target = data[:, 2, :3], data[:, 2, 3]
            else:
                if not self.four_seq:
                    seqs = data[:, :280].view(-1, 2, 140, 4)
                else: seqs = data[:, :560].view(-1, 4, 140, 4)
                lens, target = data[:, -1, :3], data[:, -1, 3]

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
            test_log_all, test_log_low, test_log_high, test_log = self._valid_epoch(epoch)
            log.update(**{'test_' + k: v for k, v in test_log_all.items()})
            log.update(**{'test_low_' + k: v for k, v in test_log_low.items()})
            log.update(**{'test_high_' + k: v for k, v in test_log_high.items()})
            log.update(**{'val_' + k: v for k, v in test_log.items()})

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
        self.test_all_metrics.reset()
        self.test_low_metrics.reset()
        self.test_high_metrics.reset()
        self.val_metrics.reset()
        with torch.no_grad():
            out_all, target_all = self._single_val_epoch(self.test_all, epoch, self.test_all_metrics)
            out_low, target_low = self._single_val_epoch(self.test_low, epoch, self.test_low_metrics)
            out_high, target_high = self._single_val_epoch(self.test_high, epoch, self.test_high_metrics)
            self._single_val_epoch(self.val_all, epoch, self.val_metrics)
            save_pred_and_target(self.log_dir, out_all, target_all, out_low, target_low, out_high, target_high)
            plot_and_save_roc(self.log_dir, (out_low, target_low, 'low'), (out_all, target_all, 'all'),
                              (out_high, target_high, 'high'))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.test_all_metrics.result(), self.test_low_metrics.result(), self.test_high_metrics.result(), self.val_metrics.result()

    def _single_val_epoch(self, val_data, epoch, metrics):
        outputs, targets = [], []
        for batch_idx, data in enumerate(val_data):
            if self.embedded:
                seqs = data[:, :2].view(-1, 200)
                lens, target = data[:, 2, :3], data[:, 2, 3]
            else:
                if not self.four_seq:
                    seqs = data[:, :280].view(-1, 2, 140, 4)
                else: seqs = data[:, :560].view(-1, 4, 140, 4)
                lens, target = data[:, -1, :3], data[:, -1, 3]

            seqs, lens, target = seqs.to(self.device), lens.to(self.device), target.to(self.device)

            output = self.model(seqs, lens)
            loss = self.criterion(output, target)

            self.writer.set_step((epoch - 1) * len(val_data) + batch_idx, 'valid')
            if 'loss' in metrics: metrics.update('loss', loss.item())
            outputs.extend(output.cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())
            for met in self.metric_ftns:
                try:
                    auc_val = met(output, target)
                    metrics.update(met.__name__, auc_val)
                except ValueError:
                    print('AUC bitching around')
                    continue
        return outputs, targets


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
