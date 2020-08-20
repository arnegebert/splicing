import argparse
import collections
import torch
import numpy as np
import data_loader as module_loader
import model.loss as module_loss
import model.metric as module_metric
import model.models as module_arch
import trainer as module_trainer
from parse_config import ConfigParser
import time
import os

# fix random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    run_id_set = 'run_id' in config
    # todo: refactor this outside of here
    if run_id_set:
        runid = config['run_id']
        os.makedirs(f'saved/{runid}', exist_ok=True)
        fname_all = f'saved/{runid}/results_all.tsv'
        if not os.path.exists(fname_all) or os.path.getsize(fname_all) == 0:
            with open(fname_all, 'w') as f:
                metrics = [metric for metric in config["logged_metrics"]]
                metrics = '\t'.join([f'{metric}' for metric in metrics])
                f.write(f'name\ttime (min)\t{metrics}\n')
        fname_concise = f'saved/{runid}/results_concise.tsv'
        if not os.path.exists(fname_all) or os.path.getsize(fname_all) == 0:
            with open(fname_concise, 'w') as f:
                cols = ['name', 'test_all (mean)', 'test_all (std)', 'test_low (mean)', 'test_low (std)',
                        'test_high (mean)', 'test_high (std)']
                cols = '\t'.join(cols)
                f.write(f'{cols}\n')

    logger = config.get_logger('train')


    folds = 9 if config['cross_validation'] else 1
    test_all, test_low, test_high = [], [], []
    for i in range(folds):
        startt = time.time()
        # setup data_loader instances
        config['data_loader']['args']['cross_validation_split'] = i
        data_loader = config.init_obj('data_loader', module_loader)
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        if i == 0: logger.info(model)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = config.init_obj('trainer', module_trainer)

        trainer.set_param(model, criterion, metrics, optimizer,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        trainer.train()

        endt = time.time()

        # Some custom logging code follows
        # todo: put this into base trainer
        test_all.append(trainer.logged_metrics["test_auc"])
        try: # this in try statement because some of my models don't have test low / high metrics
            test_low.append(trainer.logged_metrics["test_low_auc"])
            test_high.append(trainer.logged_metrics["test_high_auc"])
        except KeyError: pass

        if run_id_set: # if special run_id given, save results in central place
            runid = config['run_id']
            fname = f'{runid}.tsv'
            with open(fname, 'a') as f:
                metric_vals = [val for (key, val) in trainer.logged_metrics.items()]
                metric_vals = '\t'.join([f'{val}' for val in metric_vals])
                f.write(f'{config["name"]}\t{(endt-startt)/60:.0f}\t{metric_vals}\n')

    test_all, test_low, test_high = np.array(test_all), np.array(test_low), np.array(test_high)
    logger.info(f'Average test_all: {np.mean(test_all):.3f} +- {np.std(test_all):.3f}')
    logger.info(f'Average test_low: {np.mean(test_low):.3f} +- {np.std(test_low):.3f}')
    logger.info(f'Average test_high: {np.mean(test_high):.3f} +- {np.std(test_high):.3f}')
    if run_id_set:  # if special run_id given, save results in central place
        runid = config['run_id']
        fname = f'{runid}_concise.tsv'
        with open(fname, 'a') as f:
            f.write(f'{config["name"]}\t{np.mean(test_all):.3f}\t{np.std(test_all):.3f}\t'
                    f'{np.mean(test_low):.3f}\t{np.std(test_low):.3f}\t'
                    f'{np.mean(test_high):.3f}\t{np.std(test_high):.3f}\n')

    logger.info(f'All values test_all: {test_all}')
    logger.info(f'All values test_low: {test_low}')
    logger.info(f'All values test_high: {test_high}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-rid', '--run_id', default="test", type=str,
                      help='run_id of the experiment')
    args.add_argument('-cv', '--cross_validation', default=False, type=bool,
                      help='whether to run experiments with 9-fold cross validation or not')

    start = time.time()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
    end = time.time()
    print(f'Training took {(end-start)/60:.0f} min')