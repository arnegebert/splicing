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
import itertools

# fix random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    param_grid = config["model_parameters"]
    keys, values = zip(*param_grid.items())
    val_all, val_low, val_high = [], [], []
    # Iterate through every possible combination of hyperparameters
    logdir = config["log_directory"]
    with open(f'{logdir}/gridsearch.tsv', 'a') as f:
        f.write(f'LSTM_dim\tattn_dim\tfc_dim\tepochs\tval_all\tval_low\tval_high\n')
    for i, v in enumerate(itertools.product(*values)):
        startt = time.time()
        # Create a hyperparameter dictionary
        hyperparameters = dict(zip(keys, v))
        # 24, 24, 128: 0.77, 30 epochs
        # 24, 50 128: 0.778, 45 epochs
        # 24, 100 128: 0.762, 67 epochs
        # 24, 150 128: 0.752-0.73, 70 epochs
        # 50, 24, 128: 0.796, 73 epochs
        # got until before {'LSTM_dim': 24, 'attn_dim': 250, 'fc_dim': 128}
        if hyperparameters["LSTM_dim"] == 50 and hyperparameters["attn_dim"] == 24: continue
        if hyperparameters["LSTM_dim"] == 50 and hyperparameters["attn_dim"] == 50: continue
        print(i, hyperparameters)
        # Set model config to appropriate hyperparameters
        config["arch"]["args"].update(hyperparameters)

        config['data_loader']['args']['cross_validation_split'] = i
        data_loader = config.init_obj('data_loader', module_loader)
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)

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
        print(f'Training took {endt-startt:.3f} s')
        # not proud lol
        val_all.append(trainer.mnt_best)
        try: # this try statement because some of my models don't have val low / high metrics
            val_low.append(trainer.valid_low_metrics._data.values[1][2])
            val_high.append(trainer.valid_high_metrics._data.values[1][2])
        except AttributeError: pass
        with open(f'{logdir}/gridsearch.tsv', 'a') as f:
            f.write(f'{hyperparameters["LSTM_dim"]}\t{hyperparameters["attn_dim"]}\t{hyperparameters["fc_dim"]}\t{trainer.epochs}\t{val_all[-1]}\t{val_low[-1]}\t{val_high[-1]}\n')
    val_all, val_low, val_high = np.array(val_all), np.array(val_low), np.array(val_high)
    logger.info(f'Average val_all: {np.mean(val_all)} +- {np.std(val_all)}')
    logger.info(f'Average val_low: {np.mean(val_low)} +- {np.std(val_low)}')
    logger.info(f'Average val_high: {np.mean(val_high)} +- {np.std(val_high)}')
    logger.info(f'All observed values: {val_all}')
    logger.info(f'All observed low values: {val_low}')
    logger.info(f'All observed high values: {val_high}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/grid_search/AttnBiLSTM.json', type=str,
                      help='config file path (default: AttnBiLSTM.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-rid', '--run_id', default=None, type=str,
                      help='run_id of the experiment')


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
    print(f'Grid search took {end-start:.3f} s')