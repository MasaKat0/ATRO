import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
from collections import OrderedDict

import torch
import torchvision

from external.dada.flag_holder import FlagHolder
from external.dada.metric import MetricDict
from external.dada.io import print_metric_dict
from external.dada.io import save_model
from external.dada.logger import Logger
from external.advex_uar.advex_uar.attacks.pgd_attack import PGDAttackVariant
from external.advex_uar.advex_uar.common.pyt_common import get_step_size

from atro.vgg_variant import vgg16_variant
from atro.model import DeepLinearSvmWithRejector
from atro.loss import MaxHingeLossBinaryWithRejection
from atro.loss import WeightPenalty
from atro.data import DatasetBuilder
from atro.evaluator import Evaluator

# options
@click.command()
# model
@click.option('--dim_features', type=int, default=512)
@click.option('--dropout_prob', type=float, default=0.3)
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=1024)
@click.option('--normalize', is_flag=True, default=True)
@click.option('-t', '--binary_target_class', type=int, default=0)
# optimization
@click.option('--num_epochs', type=int, default=300)
@click.option('--lr', type=float, default=0.1, help='learning rate')
@click.option('--wd', type=float, default=5e-4, help='weight decay')
@click.option('--momentum', type=float, default=0.9)
# loss
@click.option('--cost', type=float, required=True)
# at
@click.option('--at', type=str, default=None)
@click.option('--nb_its', type=int, default=20, help='number of iterations. 20 is the same as Madry et. al., 2018.')
@click.option('--at_eps', type=float, default=0.0)
@click.option('--step_size', type=float, default=None)
@click.option('--at_norm', type=str, default=None)
# logging
@click.option('-s', '--suffix', type=str, default='')
@click.option('-l', '--log_dir', type=str, required=True)
@click.option('--ex_id', type=str, default=None)
# wandb options
@click.option("--use_wandb/--no_wandb", is_flag=True, default=True)
@click.option('--wandb_project', default='atro', help="WandB project to log to")
@click.option('--wandb_name', default='', help="WandB run name")

def main(**kwargs):
    train(**kwargs)

def train(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    FLAGS.dump(path=os.path.join(FLAGS.log_dir, 'flags{}.json'.format(FLAGS.suffix)))

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    train_dataset = dataset_builder(train=True, normalize=FLAGS.normalize, binary_classification_target=FLAGS.binary_target_class)
    val_dataset   = dataset_builder(train=False, normalize=FLAGS.normalize, binary_classification_target=FLAGS.binary_target_class)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, pin_memory=True)
    val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, FLAGS.dropout_prob).cuda()
    model = DeepLinearSvmWithRejector(features, FLAGS.dim_features, num_classes=1).cuda()
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # optimizer
    params = model.parameters() 
    optimizer = torch.optim.SGD(params, lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # loss
    MHBRLoss = MaxHingeLossBinaryWithRejection(FLAGS.cost)

    # attacker
    if FLAGS.at and FLAGS.at_eps>0:
        # get step_size
        if not FLAGS.step_size:
            FLAGS.step_size = get_step_size(FLAGS.at_eps, FLAGS.nb_its)
        assert FLAGS.step_size>=0

        # create attacker
        if FLAGS.at=='pgd':
            attacker = PGDAttackVariant(
                        FLAGS.nb_its, FLAGS.at_eps, FLAGS.step_size, dataset=FLAGS.dataset, 
                        cost=FLAGS.cost, norm=FLAGS.at_norm, num_classes=dataset_builder.num_classes, 
                        is_binary_classification = True)
        else:
            raise NotImplementedError('invalid at method.')

    # logger
    train_logger = Logger(path=os.path.join(FLAGS.log_dir,'train_log{}.csv'.format(FLAGS.suffix)), mode='train', use_wandb=False, flags=FLAGS._dict)
    val_logger   = Logger(path=os.path.join(FLAGS.log_dir,'val_log{}.csv'.format(FLAGS.suffix)), mode='val', use_wandb=FLAGS.use_wandb, flags=FLAGS._dict)

    for ep in range(FLAGS.num_epochs):
        # pre epoch
        train_metric_dict = MetricDict()
        val_metric_dict = MetricDict()

        # train
        for i, (x,t) in enumerate(train_loader):
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # adversarial attack
            if FLAGS.at and FLAGS.at_eps>0:
                model.eval()
                model.zero_grad()
                x = attacker(model, x.detach(), t.detach())

            # forward
            model.train()
            model.zero_grad()
            out_class, out_reject = model(x)

            # compute selective loss
            loss_dict = OrderedDict()
            # loss dict includes, 'A mean' / 'B mean'
            maxhinge_loss, loss_dict = MHBRLoss(out_class, out_reject, t) 
            loss_dict['maxhinge_loss'] = maxhinge_loss.detach().cpu().item()

            # regularization_loss = 0.5*WeightPenalty()(model.classifier)
            # loss_dict['regularization_loss'] = regularization_loss.detach().cpu().item()
            
            # total loss
            loss = maxhinge_loss #+ regularization_loss
            loss_dict['loss'] = loss.detach().cpu().item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric_dict.update(loss_dict)
        
        # validation
        for i, (x,t) in enumerate(val_loader):
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # adversarial attack
            if FLAGS.at and FLAGS.at_eps>0:
                model.eval()
                model.zero_grad()
                x = attacker(model, x.detach(), t.detach())

            with torch.autograd.no_grad():
                # forward
                model.eval()
                model.zero_grad()
                out_class, out_reject = model(x)

                # compute selective loss
                loss_dict = OrderedDict()
                # loss dict includes, 'A mean' / 'B mean'
                maxhinge_loss, loss_dict = MHBRLoss(out_class, out_reject, t)
                loss_dict['maxhinge_loss'] = maxhinge_loss.detach().cpu().item()
            
                # regularization_loss = 0.5*WeightPenalty()(model.classifier)
                # loss_dict['regularization_loss'] = regularization_loss.detach().cpu().item()

                # total loss
                loss = maxhinge_loss #+ regularization_loss
                loss_dict['loss'] = loss.detach().cpu().item()

                # evaluation
                evaluator = Evaluator(out_class.detach().view(-1), t.detach().view(-1), out_reject.detach().view(-1))
                loss_dict.update(evaluator())

                val_metric_dict.update(loss_dict)

        # post epoch
        # print_metric_dict(ep, FLAGS.num_epochs, train_metric_dict.avg, mode='train')
        print_metric_dict(ep, FLAGS.num_epochs, val_metric_dict.avg, mode='val')

        train_logger.log(train_metric_dict.avg, step=(ep+1))
        val_logger.log(val_metric_dict.avg, step=(ep+1))

        scheduler.step()

    # post training
    save_model(model, path=os.path.join(FLAGS.log_dir, 'weight_final{}.pth'.format(FLAGS.suffix)))


if __name__ == '__main__':
    main()