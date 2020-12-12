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
            # get list of embedding paths
            bbs = [bb for bb in os.listdir(func_path) if bb.endswith('.embed')]
            bbs.sort(key=lambda x: int(x.strip('.embed')))
            # load basic block embeddings
            x = [torch.load(os.path.join(func_path, bb)) for bb in bbs]
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
                basicblocks_d = {}
                for bb in bbs:
                    embed = torch.load(os.path.join(func_path, bb))
                    bb_id = int(bb.strip('.embed'))
                    basicblocks.append(BasicBlock(embed, d[0], bb_id))
                    basicblocks_d[bb_id] = basicblocks[-1]
                # read CFG file
                cfg = json.load(
                    open(os.path.join(func_path, 'CFG_collapsed.json')))
                # set children, parents, and edge probs for each basic block
                for basicblock in basicblocks_d.values():
                    for dest in cfg[str(basicblock.bb_id)]:
                        if len(dest) == 0 or dest[0] not in basicblocks_d:
                            continue
                        basicblock.children.append(basicblocks_d[dest[0]])
                        basicblock.children_probs.append(dest[1])
                        basicblocks_d[dest[0]].parents.append(basicblock)
                        basicblocks_d[dest[0]].parents_probs.append(dest[1])
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
