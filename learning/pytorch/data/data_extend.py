import csv
import json
import os
import random
import torch

from models.ithemal_extend import BasicBlock, Function

class DataItem:

    def __init__(self, x, y, function, code_id):
        self.x = x
        self.y = y
        self.function = function
        self.code_id = code_id


class DataExtend(object):

    def __init__(self, data_path):
        self.load_data(data_path)

    def load_data(self, data_path):
        self.train = []
        data = []
        with open(os.path.join(data_path, 'labels.csv')) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csvreader:
                data.append(row[0].split(','))
        for d in data:
            # get path to function basic blocks
            func_path = os.path.join(data_path, d[0])
            # get list of embedding pths
            bbs = [b for b in os.listdir(func_path) if b.endswith('embed')]
            # load basic block embeddings
            x = [torch.load(os.path.join(func_path, bb)) for bb in bbs]
            # create BasicBlock objects for each block
            basicblocks = []
            for bb in bbs:
                basicblocks.append(
                    BasicBlock(torch.load(os.path.join(func_path, bb)), d[0],
                               int(bb.strip('embed')[0]))
                )
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
            self.train.append(
                DataItem(x, float(d[1]), Function(basicblocks), None))
