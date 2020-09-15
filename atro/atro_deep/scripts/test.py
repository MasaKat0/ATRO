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
from external.dada.io import load_model
from external.dada.logger import Logger
from external.advex_uar.advex_uar.attacks.pgd_attack import PGDAttackVariant
from external.advex_uar.advex_uar.common.pyt_common import get_step_size

from atro.vgg_variant import vgg16_variant
from atro.model import DeepLinearSvmWithRejector
from atro.loss import MaxHingeLossWithRejection
from atro.loss import MaxHingeLossBinaryWithRejection
from atro.loss import WeightPenalty
from atro.data import DatasetBuilder
from atro.evaluator import Evaluator

# options
@click.command()
# model
@click.option('--dim_features', type=int, default=512)
@click.option('--dropout_prob', type=float, default=0.3)
@click.option('-w', '--weight', type=str, required=True, help='model weight path')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='/home/gatheluck/Scratch/selectivenet/data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=1024)
@click.option('--normalize', is_flag=True, default=True)
@click.option('-t', '--binary_target_class', type=int, default=None)
# loss
@click.option('--cost', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
# adversarial attack
@click.option('--attack', type=str, default=None)
@click.option('--nb_its', type=int, default=10)
@click.option('--step_size', type=float, default=None)
@click.option('--attack_eps', type=float, default=0.0)
@click.option('--attack_norm', type=str, default=None)


def main(**kwargs):
    test(**kwargs)

def test(**kwargs):
    """
    test model on specific cost and specific adversarial perturbation.
    """
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    assert FLAGS.nb_its>0
    assert FLAGS.attack_eps>=0

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    test_dataset   = dataset_builder(train=False, normalize=FLAGS.normalize, binary_classification_target=FLAGS.binary_target_class)
    test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, FLAGS.dropout_prob).cuda()
    if FLAGS.binary_target_class is None:
        model = DeepLinearSvmWithRejector(features, FLAGS.dim_features, dataset_builder.num_classes).cuda()
    else:
        model = DeepLinearSvmWithRejector(features, FLAGS.dim_features, 1).cuda()
    load_model(model, FLAGS.weight)

    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # loss
    if FLAGS.binary_target_class is None:
        criterion = MaxHingeLossWithRejection(FLAGS.cost)
    else:
        criterion = MaxHingeLossBinaryWithRejection(FLAGS.cost)


    # adversarial attack
    if FLAGS.attack:
        # get step_size
        if not FLAGS.step_size:
            FLAGS.step_size = get_step_size(FLAGS.attack_eps, FLAGS.nb_its)
        assert FLAGS.step_size>=0

        # create attacker
        if FLAGS.attack=='pgd':
            if FLAGS.binary_target_class is None:
                attacker = PGDAttackVariant(
                            FLAGS.nb_its, FLAGS.attack_eps, FLAGS.step_size, dataset=FLAGS.dataset, 
                            cost=FLAGS.cost, norm=FLAGS.attack_norm, num_classes=dataset_builder.num_classes,
                            is_binary_classification=False)
            else:
                attacker = PGDAttackVariant(
                            FLAGS.nb_its, FLAGS.attack_eps, FLAGS.step_size, dataset=FLAGS.dataset, 
                            cost=FLAGS.cost, norm=FLAGS.attack_norm, num_classes=dataset_builder.num_classes,
                            is_binary_classification=True)

        else:
            raise NotImplementedError('invalid attack method.')
    
    # pre epoch
    test_metric_dict = MetricDict()

    # test
    for i, (x,t) in enumerate(test_loader):
        model.eval()
        x = x.to('cuda', non_blocking=True)
        t = t.to('cuda', non_blocking=True)
        loss_dict = OrderedDict()

        # adversarial samples
        if FLAGS.attack and FLAGS.attack_eps>0:
            # create adversarial sampels
            model.zero_grad()
            x = attacker(model, x.detach(), t.detach())

        with torch.autograd.no_grad():
            model.zero_grad()
            # forward
            out_class, out_reject = model(x)
            
            # compute selective loss
            maxhinge_loss, loss_dict = criterion(out_class, out_reject, t)
            loss_dict['maxhinge_loss'] = maxhinge_loss.detach().cpu().item()

            # compute standard cross entropy loss
            # regularization_loss = WeightPenalty()(model.classifier)
            # loss_dict['regularization_loss'] = regularization_loss.detach().cpu().item()

            # total loss
            loss = maxhinge_loss #+ regularization_loss
            loss_dict['loss'] = loss.detach().cpu().item()

            # evaluation
            if FLAGS.binary_target_class is None:
                evaluator = Evaluator(out_class.detach(), t.detach(), out_reject.detach(), FLAGS.cost)
            else:
                evaluator = Evaluator(out_class.detach().view(-1), t.detach().view(-1), out_reject.detach().view(-1), FLAGS.cost)

            loss_dict.update(evaluator())

        test_metric_dict.update(loss_dict)

    # post epoch
    print_metric_dict(None, None, test_metric_dict.avg, mode='test')

    return test_metric_dict.avg

if __name__ == '__main__':
    main()