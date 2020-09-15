import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import re
import click
import uuid
import glob

from collections import OrderedDict

from external.dada.flag_holder import FlagHolder
from external.dada.logger import Logger
from scripts.test import test

EPS = {
    'pgd_linf': [0, 1, 2, 4, 8, 16],
    'pgd_l2':   [0, 40, 80, 160, 320, 640],
}


# options
@click.command()
# target
@click.option('-t', '--target_dir', type=str, required=True)
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('--binary_target_class', type=int, default=None)
@click.option('--cost', type=float, default=None)
# adversarial training
@click.option('--at', type=str, default=None)
@click.option('--at_norm', type=str, default=None)
# adversarial attack
@click.option('--attack', type=str, required=True)
@click.option('--attack_norm', type=str, required=True)
@click.option('--nb_its', type=int, default=10)
# log
@click.option('-s', '--suffix', type=str, default='')

def main(**kwargs):
    test_multi_adv(**kwargs)

def parse_weight_basename(weight_basename):
    ret_dict = dict()

    # remove ext
    basename, ext = os.path.splitext(weight_basename)

    # extract cost and else
    # \d : any single number
    # \w : any single number or charactor
    # .  : any single charactor
    # +  : sequence more than one time
    # *  : sequence more than zero time

    # 'weight_final_cost_{cost}_{else}'
    pattern = r'weight_final_cost-(\d.\d+)_(.*)'
    result = re.match(pattern, basename)

    ret_dict['cost'] = float(result.group(1))

    at_info = result.group(2)

    if at_info == 'std':
        ret_dict['at'] = None
        ret_dict['at_norm'] = None
        ret_dict['at_eps'] = 0.0
    else:
        # '{at}-{at_norm}_eps-{at_eps}' 
        pattern = r'(\w+)-(\w+)_eps-(\d+)'
        result = re.match(pattern, at_info)

        ret_dict['at'] = result.group(1)
        ret_dict['at_norm'] = result.group(2)
        ret_dict['at_eps'] = float(result.group(3))
        
    return ret_dict

def test_multi_adv(**kwargs):
    """
    this script loads all 'weight_final_{something}.pth' files which exisits under 'kwargs.target_dir' and execute test.
    if there is exactly same file, the result becomes the mean of them.
    the results are saved as csv file.

    'target_dir' should be like follow
    (.pth file name should be "weight_final_cost_{}") 

    ~/target_dir/XXXX/weight_final_cost_0.10_pgd-linf_eps-0.pth
                     ...
                     /weight_final_cost_0.10_pgd-linf_eps-8.pth
                     /weight_final_cost_0.10_pgd-linf_eps-16.pth
                     ...
                /YYYY/weight_final_cost_0.10_pgd-linf_eps-0.pth
                     ...
                     /weight_final_cost_0.10_pgd-linf_eps-8.pth
                     /weight_final_cost_0.10_pgd-linf_eps-16.pth
                     ...
    """
    # flags
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # specify target weight path
    run_dir  = '../scripts'
    target_path = os.path.join(FLAGS.target_dir, '**/weight_final*.pth')
    weight_paths = sorted(glob.glob(target_path, recursive=True), key=lambda x: os.path.basename(x))

    if FLAGS.cost is not None:
        weight_paths = [wpath for wpath in weight_paths if 'cost-{cost:0.2f}'.format(cost=FLAGS.cost) in wpath]
    if FLAGS.at is not None:
        weight_paths = [wpath for wpath in weight_paths if '{at}-{at_norm}'.format(at=FLAGS.at, at_norm=FLAGS.at_norm) in wpath]


    log_path = os.path.join(FLAGS.target_dir, 'test{}.csv'.format(FLAGS.suffix))

    # logging
    logger = Logger(path=log_path, mode='test', use_wandb=False, flags=FLAGS)

    # get epses
    key = FLAGS.attack + '_' + FLAGS.attack_norm
    attack_epses = EPS[key]

    for weight_path in weight_paths:
        for attack_eps in attack_epses:

            # parse basename
            basename = os.path.basename(weight_path)
            ret_dict = parse_weight_basename(basename)

            # keyword args for test function
            # variable args
            kw_args = {}
            kw_args['weight'] = weight_path
            kw_args['dataset'] = FLAGS.dataset
            kw_args['dataroot'] = FLAGS.dataroot
            kw_args['binary_target_class'] = FLAGS.binary_target_class
            kw_args['cost'] = ret_dict['cost']
            kw_args['attack'] = FLAGS.attack
            kw_args['nb_its'] = FLAGS.nb_its
            kw_args['step_size'] = None
            kw_args['attack_eps'] = attack_eps
            kw_args['attack_norm'] = FLAGS.attack_norm

            # default args
            kw_args['dim_features'] = 512
            kw_args['dropout_prob'] = 0.3
            kw_args['num_workers'] = 8
            kw_args['batch_size'] = 128
            kw_args['normalize'] = True
            kw_args['alpha'] = 0.5
            
            # run test
            out_dict = test(**kw_args)

            metric_dict = OrderedDict()
            metric_dict['cost'] = ret_dict['cost']
            metric_dict['binary_target_class'] = FLAGS.binary_target_class
            # at
            metric_dict['at'] = ret_dict['at']
            metric_dict['at_norm'] = ret_dict['at_norm']
            metric_dict['at_eps'] = ret_dict['at_eps']
            # attack
            metric_dict['attack'] = FLAGS.attack 
            metric_dict['attack_norm'] = FLAGS.attack_norm
            metric_dict['attack_eps'] = attack_eps
            # path
            metric_dict['path'] = weight_path
            metric_dict.update(out_dict)

            # log
            logger.log(metric_dict)

if __name__ == '__main__':
    main()