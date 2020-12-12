import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import argparse
import datetime
import random
import torch

import models.graph_models as md
import models.train as tr
import training
from data.data_extend import DataExtend
from experiments.experiment import Experiment
from ithemal_utils import *
from models.ithemal_extend import RNNExtend, GraphNN


def get_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--embed-size', help='The size of embedding to use (default: 256)', default=256, type=int)
    parser.add_argument('--hidden-size', help='The size of hidden layer to use (default: 256)', default=256, type=int)
    parser.add_argument('--no-mem', help='Remove all instructions with memory', default=False, action='store_true')

    parser.add_argument('--use-rnn', action='store_true', default=False)
    parser.add_argument('--no-residual', default=False, action='store_true', help='Don\'t use a residual model in Ithemal')
    parser.add_argument('--no-dag-rnn', default=False, action='store_true', help='Don\'t use the DAG-RNN model in Ithemal')

    parser.add_argument('--use-scaling', action='store_true', help='Whether to scale model output', default=False)
    parser.add_argument('--scale-amount', type=float, default=1000., help='Amount to scale by')
    #

    sp = parser.add_subparsers(dest='subparser')

    # train
    train = sp.add_parser('train', help='Train an ithemal model')
    train.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    train.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    train.add_argument('--load-file', help='Start by loading the provided model')
    train.add_argument('--test', action='store_true', help='Test mode', default=False)

    train.add_argument('--batch-size', type=int, default=4, help='The batch size to use in train')
    train.add_argument('--epochs', type=int, default=3, help='Number of epochs to run for')
    train.add_argument('--trainers', type=int, default=1, help='Number of trainer processes to use')
    train.add_argument('--threads', type=int,  default=4, help='Total number of PyTorch threads to create per trainer')
    train.add_argument('--decay-trainers', action='store_true', default=False, help='Decay the number of trainers at the end of each epoch')
    train.add_argument('--weight-decay', type=float, default=0, help='Coefficient of weight decay (L2 regularization) on model')
    train.add_argument('--initial-lr', type=float, default=0.1, help='Initial learning rate')
    train.add_argument('--decay-lr', action='store_true', default=False, help='Decay the learning rate at the end of each epoch')
    train.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter for SGD')
    train.add_argument('--nesterov', action='store_true', default=False, help='Use Nesterov momentum')
    train.add_argument('--weird-lr', action='store_true', default=False, help='Use unusual LR schedule')
    train.add_argument('--lr-decay-rate', default=1.2, help='LR division rate', type=float)
    #

    # GraphNN
    dag_nonlinearity_group = parser.add_mutually_exclusive_group()
    dag_nonlinearity_group.add_argument('--dag-relu-nonlinearity', action='store_const', const=md.NonlinearityType.RELU, dest='dag_nonlinearity')
    dag_nonlinearity_group.add_argument('--dag-tanh-nonlinearity', action='store_const', const=md.NonlinearityType.TANH, dest='dag_nonlinearity')
    dag_nonlinearity_group.add_argument('--dag-sigmoid-nonlinearity', action='store_const', const=md.NonlinearityType.SIGMOID, dest='dag_nonlinearity')
    parser.set_defaults(dag_nonlinearity=None)
    parser.add_argument('--dag-nonlinearity-width', help='The width of the final nonlinearity (default: 128)', default=128, type=int)
    parser.add_argument('--dag-nonlinear-before-max', action='store_true', default=False)

    dag_reduction_group = parser.add_mutually_exclusive_group()
    dag_reduction_group.add_argument('--dag-add-reduction', action='store_const', const=md.ReductionType.ADD, dest='dag_reduction')
    dag_reduction_group.add_argument('--dag-max-reduction', action='store_const', const=md.ReductionType.MAX, dest='dag_reduction')
    dag_reduction_group.add_argument('--dag-mean-reduction', action='store_const', const=md.ReductionType.MEAN, dest='dag_reduction')
    dag_reduction_group.add_argument('--dag-attention-reduction', action='store_const', const=md.ReductionType.ATTENTION, dest='dag_reduction')
    parser.set_defaults(dag_reduction=md.ReductionType.MAX)
    #

    # optimizer
    optimizer_group = train.add_mutually_exclusive_group()
    optimizer_group.add_argument('--adam-private', action='store_const', const=tr.OptimizerType.ADAM_PRIVATE, dest='optimizer', help='Use Adam with private moments',
                                 default=tr.OptimizerType.ADAM_PRIVATE)
    optimizer_group.add_argument('--adam-shared', action='store_const', const=tr.OptimizerType.ADAM_SHARED, dest='optimizer', help='Use Adam with shared moments')
    optimizer_group.add_argument('--sgd', action='store_const', const=tr.OptimizerType.SGD, dest='optimizer', help='Use SGD')
    #

    return parser


def get_base_parameters(args):
    base_params = BaseParameters(
        data=args.data,
        embed_mode=None,
        embed_file=None,
        random_edge_freq=None,
        predict_log=None,
        no_residual=args.no_residual,
        no_dag_rnn=args.no_dag_rnn,
        dag_reduction=args.dag_reduction,
        edge_ablation_types=None,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        linear_embeddings=None,
        use_rnn=args.use_rnn,
        rnn_type=None,
        rnn_hierarchy_type=None,
        rnn_connect_tokens=None,
        rnn_skip_connections=None,
        rnn_learn_init=None,
        no_mem=args.no_mem,
        linear_dependencies=None,
        flat_dependencies=None,
        dag_nonlinearity=args.dag_nonlinearity,
        dag_nonlinearity_width=args.dag_nonlinearity_width,
        dag_nonlinear_before_max=args.dag_nonlinear_before_max,
    )
    return base_params


def get_train_parameters(args):
    train_params = TrainParameters(
        experiment_name=args.experiment_name,
        experiment_time=args.experiment_time,
        load_file=args.load_file,
        batch_size=args.batch_size,
        trainers=args.trainers,
        threads=args.threads,
        decay_trainers=args.decay_trainers,
        weight_decay=args.weight_decay,
        initial_lr=args.initial_lr,
        decay_lr=args.decay_lr,
        epochs=args.epochs,
        split=None,
        optimizer=args.optimizer,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weird_lr=args.weird_lr,
        lr_decay_rate=args.lr_decay_rate,
    )
    return train_params


def load_data(params):
    # type: (BaseParameters) -> dt.DataCost
    # TODO (thomaseh): finish dataloader
    data = DataExtend(params.data, params.use_rnn)
    # assert False

    return data


def load_model(params, args):
    # type: (BaseParameters) -> md.AbstractGraphModule
    if params.use_rnn:
        rnn_params = md.RnnParameters(
            embedding_size=params.embed_size,
            hidden_size=params.hidden_size,
            num_classes=1,
            connect_tokens=False,           # NOT USED
            skip_connections=False,         # NOT USED
            hierarchy_type='MULTISCALE',    # NOT USED
            rnn_type='LSTM',                # NOT USED
            learn_init=True,                # NOT USED
        )
        model = RNNExtend(rnn_params, args)
    else:
        model = GraphNN(
            embedding_size=params.embed_size, hidden_size=params.hidden_size,
            num_classes=1, use_residual=not params.no_residual,
            use_dag_rnn=not params.no_dag_rnn, reduction=params.dag_reduction,
            nonlinear_type=params.dag_nonlinearity,
            nonlinear_width=params.dag_nonlinearity_width,
            nonlinear_before_max=params.dag_nonlinear_before_max,
        )

    return model


def get_save_directory(exp_name, exp_time):
    now = datetime.datetime.now()
    timestamp = '%d%02d%02d%02d%02d%02d' % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    save_path = os.path.join(
        'learning/pytorch/saved', exp_name, exp_time, 'checkpoints', timestamp)
    return save_path


def train(data, model, base_params, train_params, save_dir):
    trainer = training.load_trainer(base_params, train_params, model, data)
    expt = Experiment(
        train_params.experiment_name, train_params.experiment_time,
        base_params.data)
    loss_reporter = training.LossReporter(expt, len(data.train), trainer)
    def report_loss_fn(msg):
        loss_reporter.report_items(msg.n_items, msg.loss)

    for epoch_no in range(train_params.epochs):
        loss_reporter.start_epoch(epoch_no + 1, 0)
        random.shuffle(data.train)
        trainer.train(report_loss_fn=report_loss_fn)
        loss_reporter.report()
        # 583 set how often to save models
        if epoch_no % 50 == 0:
            save_file = os.path.join(save_dir, 'epoch_%03d.mdl' % (epoch_no+1,))
            trainer.save_checkpoint(epoch_no, -1, save_file)
    save_file = os.path.join(save_dir, 'epoch_final.mdl')
    trainer.save_checkpoint(epoch_no, -1, save_file)

    return trainer


def test(trainer, save_dir):
    trainer.validate(os.path.join(save_dir, 'results.csv'))


def main():
    # type: () -> None
    args = get_parser().parse_args()

    base_params = get_base_parameters(args)

    if args.subparser == 'train':
        train_params = get_train_parameters(args)

        # load data and model
        print('Loading data and setting up model...')
        data = load_data(base_params)
        model = load_model(base_params, args)

        if not args.test:
            # train
            print('Training...')
            save_dir = get_save_directory(
                train_params.experiment_name, train_params.experiment_time)
            trainer = train(data, model, base_params, train_params, save_dir)
        else:
            trainer = training.load_trainer(base_params, train_params, model, data)
            trainer.load_checkpoint(args.load_file)
            save_dir = '/'.join(args.load_file.split('/')[:-1])

        # test
        print('Testing...')
        test(trainer, save_dir)
    else:
        raise ValueError('Unknown mode "{}"'.format(args.subparser))

if __name__ == '__main__':
    main()
