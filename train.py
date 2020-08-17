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


# fix random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    iters = 10 if config['cross_validation'] else 1
    test_all, test_low, test_high = [], [], []
    for i in range(iters):
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
        # not proud lol
        # todo: find best or adapt so that best is found
        raise Exception('Best test monitor functionality not implemented yet')
        test_all.append(trainer.mnt_best)
        try: # this try statement because some of my models don't have test low / high metrics
            test_low.append(trainer.test_low_metrics._data.values[0][2])
            test_high.append(trainer.test_high_metrics._data.values[0][2])
        except AttributeError: pass
    test_all, test_low, test_high = np.array(test_all), np.array(test_low), np.array(test_high)
    logger.info(f'Average test_all: {np.mean(test_all)} +- {np.std(test_all)}')
    logger.info(f'Average test_low: {np.mean(test_low)} +- {np.std(test_low)}')
    logger.info(f'Average test_high: {np.mean(test_high)} +- {np.std(test_high)}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-rid', '--run_id', default=None, type=str,
                      help='run_id of the experiment')
    args.add_argument('-cv', '--cross_validation', default=False, type=bool,
                      help='whether to run experiments with 10-fold cross validation or not')

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
    print(f'Training took {end-start:.3f} s')