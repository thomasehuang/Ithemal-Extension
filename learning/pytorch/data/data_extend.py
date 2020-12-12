import csv
import json
import os
import random
import torch
import numpy as np

from models.ithemal_extend import BasicBlock, Function

class DataItem:

    def __init__(self, x, y, function, code_id):
        self.x = x
        self.y = y
        self.function = function
        self.code_id = code_id


class DataExtend(object):

    def __init__(self, data_path, use_rnn=True):
        self.load_data(data_path, use_rnn)

    def load_data(self, data_path, use_rnn):
        self.data, self.train, self.test = [], [], []
        data = []
        with open(os.path.join(data_path, 'labels.csv')) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            first_flag = True
            for row in csvreader:
                if first_flag:
                    first_flag = False
                    continue
                row = row[0].split(',')
                data.append(row)
        for d in data:
            # get path to function basic blocks
            func_path = os.path.join(data_path, d[0])
            # get number of basic blocks
            n_bb = len([b for b in os.listdir(func_path) if b.endswith('o')])
            # get list of embedding paths
            bbs = []
            for bb_id in range(n_bb):
                embed = '%d.embed' % (bb_id,)
                embed = embed if embed in os.listdir(func_path) \
                            else '%d.none' % (bb_id,)
                bbs.append(embed)
            # load basic block embeddings
            x = [torch.load(os.path.join(func_path, bb)) \
                    for bb in bbs if not bb.endswith('none')]
            if len(x) == 0:
                continue
            # create DataItem
            if use_rnn:
                # regular RNN
                self.data.append(
                    DataItem(x, float(d[1]), Function([], d[0]), None))
            else:
                # GraphNN
                # create BasicBlock objects for each block
                basicblocks = []
                for bb in bbs:
                    embed = torch.load(os.path.join(func_path, bb)) \
                        if not bb.endswith('none') else None
                    bb_id = int(bb.strip('.')[0])
                    basicblocks.append(BasicBlock(embed, d[0], bb_id))
                # read CFG file
                cfg = json.load(open(os.path.join(func_path, 'CFG.json')))
                # set children, parents, and edge probs for each basic block
                for i, basicblock in enumerate(basicblocks):
                    if str(i) not in cfg:
                        continue
                    for dest in cfg[str(i)]:
                        basicblock.children.append(basicblocks[dest[0]])
                        basicblock.children_probs.append(dest[1])
                        basicblocks[dest[0]].parents.append(basicblock)
                        basicblocks[dest[0]].parents_probs.append(dest[1])
                self.data.append(
                    DataItem(x, float(d[1]), Function(basicblocks, d[0]), None))
        # split data into train and val
        idx = int(len(self.data) * 0.8)
        self.train = self.data[:idx]
        # apply transformation to labels
        self.get_train_stats()
        for ex in self.train:
            ex.y = self.transform_label(ex.y)

        self.test = self.data[idx:]

    def get_train_stats(self):
        train_ys = [ex.y for ex in self.train]
        self.train_mean = np.mean(train_ys)
        self.train_std = np.std(train_ys)

    def transform_label(self, y):
        # return (y - self.train_mean) / self.train_std
        return np.log(y)

    def inverse_label_transform(self, y):
        # return y * self.train_std + self.train_mean
        return torch.exp(y)
