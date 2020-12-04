import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import argparse
import torch

import models.train as tr
import training
from ithemal_utils import *


def load_data(params):
    # type: (BaseParameters) -> dt.DataCost
    # TODO (thomaseh): finish dataloader
    data = None
    assert False

    return data


def load_model(params):
    # type: (BaseParameters) -> md.AbstractGraphModule
    rnn_params = RnnParameters(
        embedding_size=params.embed_size,
        hidden_size=params.hidden_size,
        num_classes=1,
        connect_tokens=False,           # NOT USED
        skip_connections=False,         # NOT USED
        hierarchy_type='MULTISCALE',    # NOT USED
        rnn_type='LSTM',                # NOT USED
        learn_init=True,                # NOT USED
    )
    model = RNNExtend(rnn_params)

    return model


def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--embed-size', help='The size of embedding to use (default: 256)', default=256, type=int)
    parser.add_argument('--hidden-size', help='The size of hidden layer to use (default: 256)', default=256, type=int)
    parser.add_argument('--no-mem', help='Remove all instructions with memory', default=False, action='store_true')

    sp = parser.add_subparsers(dest='subparser')

    train = sp.add_parser('train', help='Train an ithemal model')
    train.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    train.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    train.add_argument('--load-file', help='Start by loading the provided model')

    train.add_argument('--batch-size', type=int, default=4, help='The batch size to use in train')
    train.add_argument('--epochs', type=int, default=3, help='Number of epochs to run for')
    train.add_argument('--trainers', type=int, default=4, help='Number of trainer processes to use')
    train.add_argument('--threads', type=int,  default=4, help='Total number of PyTorch threads to create per trainer')
    train.add_argument('--decay-trainers', action='store_true', default=False, help='Decay the number of trainers at the end of each epoch')
    train.add_argument('--weight-decay', type=float, default=0, help='Coefficient of weight decay (L2 regularization) on model')
    train.add_argument('--initial-lr', type=float, default=0.1, help='Initial learning rate')
    train.add_argument('--decay-lr', action='store_true', default=False, help='Decay the learning rate at the end of each epoch')
    train.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter for SGD')
    train.add_argument('--nesterov', action='store_true', default=False, help='Use Nesterov momentum')
    train.add_argument('--weird-lr', action='store_true', default=False, help='Use unusual LR schedule')
    train.add_argument('--lr-decay-rate', default=1.2, help='LR division rate', type=float)

    optimizer_group = train.add_mutually_exclusive_group()
    optimizer_group.add_argument('--adam-private', action='store_const', const=tr.OptimizerType.ADAM_PRIVATE, dest='optimizer', help='Use Adam with private moments',
                                 default=tr.OptimizerType.ADAM_PRIVATE)
    optimizer_group.add_argument('--adam-shared', action='store_const', const=tr.OptimizerType.ADAM_SHARED, dest='optimizer', help='Use Adam with shared moments')
    optimizer_group.add_argument('--sgd', action='store_const', const=tr.OptimizerType.SGD, dest='optimizer', help='Use SGD')

    args = parser.parse_args()

    base_params = BaseParameters(
        data=args.data,
        embed_mode=None,
        embed_file=None,
        random_edge_freq=None,
        predict_log=None,
        no_residual=None,
        no_dag_rnn=None,
        dag_reduction=None,
        edge_ablation_types=None,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        linear_embeddings=None,
        use_rnn=None,
        rnn_type=None,
        rnn_hierarchy_type=None,
        rnn_connect_tokens=None,
        rnn_skip_connections=None,
        rnn_learn_init=None,
        no_mem=args.no_mem,
        linear_dependencies=None,
        flat_dependencies=None,
        dag_nonlinearity=None,
        dag_nonlinearity_width=None,
        dag_nonlinear_before_max=None,
    )

    if args.subparser == 'train':
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
        # training.run_training_coordinator(base_params, train_params)
    else:
        raise ValueError('Unknown mode "{}"'.format(args.subparser))

if __name__ == '__main__':
    main()