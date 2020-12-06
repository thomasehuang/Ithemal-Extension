import csv
import os
import random
import torch


class DataItem:

    def __init__(self, x, y, block, code_id):
        self.x = x
        self.y = y
        self.block = block
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
            func_path = os.path.join(data_path, d[0])
            bbs = [b for b in os.listdir(func_path) if b.endswith('embed')]
            x = [torch.load(os.path.join(func_path, bb)) for bb in bbs]
            self.train.append(DataItem(x, float(d[1]), None, None))
