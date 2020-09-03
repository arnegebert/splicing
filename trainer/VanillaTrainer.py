import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, confusion_matrix

from base import BaseTrainer
from model.metric import auc_single
from utils import inf_loop, MetricTracker, save_pred_and_target, plot_and_save_roc

class VanillaTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, embedded=False, attention=False):
        self.embedded = embedded
        self.attention = attention
        # could potentially add cross-validation support here

    def set_param(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, four_seq=False):
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
        self.test_all, self.test_low, self.test_high, self.val_all = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.four_seq = four_seq

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_all_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
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
            seqs, lens, target = self.convert_to_model_input_format(data)
            seqs, lens, target = seqs.to(self.device), lens.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(seqs, lens)
            if self.attention:
                pred, attn_ws = pred

            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

            # logs that this training step has been taken at current time
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                try:
                    auc_val = met(pred, target)
                    self.train_metrics.update(met.__name__, auc_val)
                except ValueError as e:
                    print(e)
                    # print('AUC bitching around for train metrics')
                    continue

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        test_log_all, test_log_low, test_log_high, val_log = self._valid_epoch(epoch)
        log.update(**{'test_' + k: v for k, v in test_log_all.items()})
        log.update(**{'test_low_' + k: v for k, v in test_log_low.items()})
        log.update(**{'test_high_' + k: v for k, v in test_log_high.items()})
        log.update(**{'val_' + k: v for k, v in val_log.items()})

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
                pred_target_all = self._single_val_epoch(self.test_all, epoch, self.test_all_metrics)
                pred_target_low = self._single_val_epoch(self.test_low, epoch, self.test_low_metrics)
                pred_target_high = self._single_val_epoch(self.test_high, epoch, self.test_high_metrics)
                pred_target_val = self._single_val_epoch(self.val_all, epoch, self.val_metrics)
                # todo: make this cleaner; assumes that mnt_best tracks auc
                try:
                    if auc_single(pred_target_val) >= self.mnt_best:
                        self.auc_f1_metric_evaluation_and_visualization(pred_target_all, pred_target_low, pred_target_high)
                except ValueError:
                    pass


        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.test_all_metrics.result(), self.test_low_metrics.result(), self.test_high_metrics.result(), \
               self.val_metrics.result()

    def auc_f1_metric_evaluation_and_visualization(self, pred_target_all, pred_target_low, pred_target_high):
        save_pred_and_target(self.log_dir, pred_target_all, pred_target_low, pred_target_high)
        plot_and_save_roc(self.log_dir, (pred_target_low, 'low'), (pred_target_all, 'all'),
                          (pred_target_high, 'high'))
        pred_all, target_all = pred_target_all
        # taken from https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        precision, recall, thresholds = precision_recall_curve(target_all, pred_all)
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.nanargmax(fscore)
        tp, fp, fn, tn = confusion_matrix(target_all, pred_all >= thresholds[ix]).ravel()
        self.logger.info(f'Best Threshold: {thresholds[ix]}, F-Score={fscore[ix]:.3f}')
        self.logger.info(f'Corresponding precision: {precision[ix]:.3f}, recall: {recall[ix]:.3f}')
        # the * indicates that we fixed the bug were the namings of TP and TN were switched in the code
        # generating the log file
        self.logger.info(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn} *')



    def _single_val_epoch(self, val_data, epoch, metrics):
        predictions, targets = [], []
        attn_ws_b = []
        datab = []
        for batch_idx, data in enumerate(val_data):
            seqs, lens, target = self.convert_to_model_input_format(data)
            seqs, lens, target = seqs.to(self.device), lens.to(self.device), target.to(self.device)

            pred = self.model(seqs, lens)
            if self.attention:
                pred, attn_ws = pred
                attn_ws = attn_ws.data.cpu().numpy()
                attn_ws_b.append(attn_ws)
                databs = data.data.cpu().numpy()
                datab.append(databs)

            loss = self.criterion(pred, target)

            self.writer.set_step((epoch - 1) * len(val_data) + batch_idx, 'valid')
            if 'loss' in metrics: metrics.update('loss', loss.item())
            predictions.append(pred.data)
            targets.append(target.data)
            for met in self.metric_ftns:
                try:
                    auc_val = met(pred, target)
                    metrics.update(met.__name__, auc_val)
                except ValueError as e:
                    print(e)
                    # print('AUC bitching around')
                    continue
        if self.attention and metrics == self.test_all_metrics:
            attn_ws_b = np.concatenate(attn_ws_b, axis=0)
            for attnw in sum(np.mean(attn_ws_b[:, :140], axis=0)):
                print(f'{attnw:.2f}')
            if epoch == 50:
                np.save(f'test_all_data.npy', np.concatenate(datab, axis=0))
            np.save(f'attn_ws_{epoch}.npy', attn_ws_b)
        del attn_ws_b
        predictions, targets = torch.cat(predictions, dim=0).cpu().numpy(), torch.cat(targets, dim=0).cpu().numpy()
        return predictions, targets

    def convert_to_model_input_format(self, data):
        if self.embedded:
            if not self.four_seq:
                seqs = data[:, :2].view(-1, 200)
                lens, target = data[:, 2, :3], data[:, 2, 3]
            else:
                seqs = data[:, :4].view(-1, 400)
                lens, target = data[:, 4, :3], data[:, 4, 3]
        else:
            if not self.four_seq:
                seqs = data[:, :280].view(-1, 2, 140, 4)
            else:
                seqs = data[:, :560].view(-1, 4, 140, 4)
            lens, target = data[:, -1, :3], data[:, -1, 3]
        return seqs, lens, target

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
