import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import importlib
import pandas
from datetime import datetime
from collections import OrderedDict

from external.dada.misc import get_time_stamp

class Logger(object):
    """
    logger class for logging info to csv files.
    """
    MODES = set(['train', 'val', 'test'])

    def __init__(self, path:str, mode:str, use_wandb=True, flags=None):
        if mode in self.MODES:
            self.mode = mode
        else: 
            raise ValueError('mode is invalid')

        self.path = path
        self.use_wandb = use_wandb
        self.df = pandas.DataFrame()
        self._save()

        self.row_idx = 0

        # wandb
        if self.use_wandb:
            globals()['wandb'] = importlib.import_module('wandb')
            self._configure_wandb(use_wandb, flags['wandb_project'])
            self._init_wandb(flags)

    def log(self, log_dict, step=None):
        """
            step | time stamp | val01 | val02 | val03
        0
        -
        1
        -

        Appending new column is much easier than appending new row.


        Args:
            log_dict:
            step:
        """
        self.df = pandas.read_csv(self.path, index_col=0)
        time_stamp = get_time_stamp()

        # create data dict for adding new data to csv file
        datadict = OrderedDict()
        if (self.mode=='train') or (self.mode=='val'):
            datadict['step'] = int(step)
        datadict['time stamp'] = time_stamp

        for k,v in log_dict.items():
            datadict[k] = v

        new_df = pandas.DataFrame(datadict, index=[self.row_idx])
        self.df = self.df.append(new_df, sort=False)

        self._save()

        # wandb
        if self.use_wandb: wandb.log(datadict, step=step)

        self.row_idx += 1

    def _save(self):
        self.df.to_csv(self.path)

    def _configure_wandb(self, use_wandb, project_name):
        if os.getenv('WANDB_API_KEY') is None or not use_wandb:
            os.environ['WANDB_MODE'] = 'dryrun'
        else:
            os.environ['WANDB_PROJECT'] = project_name

    def _init_wandb(self, flag_dict):
        if flag_dict['wandb_name']:
            out = wandb.init(dir=flag_dict['log_dir'], name=flag_dict['wandb_name'])
        else:
            out = wandb.init(dir=flag_dict['log_dir'])
        print('wandb initialized', out, wandb.config)
        for k,v in flag_dict.items():
            setattr(wandb.config, k, v)
        setattr(wandb.config, 'run_id', wandb.run.id)

if __name__ == '__main__':
    log_path_root = '/home/gatheluck/Scratch/selectivenet/logs'
    log_basename = 'log_test_'+get_time_stamp('short')
    log_path = os.path.join(log_path_root, log_basename)

    logger = Logger(log_path)

    log_dict  = {'loss01':1.0, 'loss02':2.0}
    log_dict_ = {'loss01':1.0, 'loss03':3.0}
    logger.log(log_dict, 1)
    logger.log(log_dict, 2)
    logger.log(log_dict, 3)
    logger.log(log_dict_, 4)
    