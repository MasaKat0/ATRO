import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import subprocess
import click
import uuid

from external.dada.flag_holder import FlagHolder
from external.abci_util.script_generator import generate_script

# options
@click.command()
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
# optimization
@click.option('--num_epochs', type=int, default=200)
@click.option('-N', '--batch_size', type=int, default=1024)
# logging
@click.option('-l', '--log_dir', type=str, required=True)
@click.option('--ex_id', type=str, default=uuid.uuid4().hex, help='id of the experiments')
# selective loss
@click.option('--cost', type=float, default=None)
# at
@click.option('--nb_its', type=int, default=20, help='number of iterations. 20 is the same as Madry et. al., 2018.')
@click.option('--at_eps', type=float, default=None)
@click.option('--at_norm', type=str, default=None)
# option for abci
@click.option('--hour', type=int, default=24)
@click.option('--script_root', type=str, required=True)
@click.option('--run_dir', type=str, required=True)
@click.option('--abci_log_dir', type=str, default='~/abci_log')
@click.option('--user', type=str, required=True)
@click.option('--env', type=str, required=True)
# wandb
@click.option('--wandb_api_key', type=str, default='')
@click.option("--use_wandb/--no_wandb", is_flag=True, default=True)
@click.option('--wandb_project', default='atro', help="WandB project to log to")

def main(**kwargs):
    train_multi(**kwargs)

def train_multi(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # create script output dir
    script_dir = os.path.join(FLAGS.script_root, FLAGS.ex_id)
    os.makedirs(script_dir, exist_ok=True)

    costs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] if not FLAGS.cost else [FLAGS.cost]
    ats = ['pgd']
    at_norms = ['linf', 'l2'] if not FLAGS.at_norm else [FLAGS.at_norm]

    EPS = {
        'pgd-linf': [0., 1., 2., 4., 8., 16.],
        'pgd-l2':   [0., 40., 80., 160., 320., 640.],
    }

    for cost in sorted(costs):
        for at in ats:
            for at_norm in at_norms:
                key = at+'-'+at_norm
                epses = EPS[key] if FLAGS.at_eps is None else [FLAGS.at_eps]

                for at_eps in epses:

                    suffix = '_cost-{cost:0.2f}_{at}-{at_norm}_eps-{at_eps:0.1f}'.format(
                        cost=cost, at=at, at_norm=at_norm, at_eps=at_eps) 

                    log_dir = os.path.join(FLAGS.log_dir, FLAGS.ex_id)
                    os.makedirs(log_dir, exist_ok=True)

                    cmd = 'python train_binary.py \
                          -d {dataset} \
                          --dataroot {dataroot} \
                          --num_epochs {num_epochs} \
                          --batch_size {batch_size} \
                          --cost {cost} \
                          --at {at} \
                          --nb_its {nb_its} \
                          --at_eps {at_eps} \
                          --at_norm {at_norm} \
                          -s {suffix} \
                          -l {log_dir} \
                          {use_wandb} \
                          --wandb_project {wandb_project} \
                          --wandb_name {wandb_name}'.format(
                            dataset=FLAGS.dataset,
                            dataroot=FLAGS.dataroot,
                            num_epochs=FLAGS.num_epochs,
                            batch_size=FLAGS.batch_size,
                            cost=cost,
                            at=at,
                            nb_its=FLAGS.nb_its,
                            at_eps=at_eps,
                            at_norm=at_norm,
                            suffix=suffix,
                            log_dir=log_dir,
                            use_wandb='--use_wandb' if FLAGS.use_wandb else '',
                            wandb_project=FLAGS.wandb_project,
                            wandb_name=suffix.lstrip('_'))

                    script_basename = suffix.lstrip('_')+'.sh'
                    script_path = os.path.join(script_dir, script_basename)
                    generate_script(cmd, script_path, FLAGS.run_dir, FLAGS.abci_log_dir, FLAGS.ex_id, FLAGS.user, FLAGS.env, hour=FLAGS.hour, wandb_api_key=FLAGS.wandb_api_key)

if __name__ == '__main__':
    main()