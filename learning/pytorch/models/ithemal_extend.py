import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

from enum import Enum, unique
import torch
import torch.nn as nn
import torch.nn.functional as F
import common_libs.utilities as ut
import data.data_cost as dt
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple
from . import model_utils

from graph_models import *


class RNNExtend(AbstractGraphModule):

    def __init__(self, params):
        # type: (RnnParameters) -> None
        super(RNNExtend, self).__init__(params.embedding_size, params.hidden_size, params.num_classes)

        self.params = params

        # assuming LSTM for now
        self.bb_rnn = nn.LSTM(self.hidden_size, self.hidden_size)

        self._bb_init = self.rnn_init_hidden()

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def rnn_init_hidden(self):
        # type: () -> Union[Tuple[nn.Parameter, nn.Parameter], nn.Parameter]
       return self.init_hidden()

    def get_bb_init(self):
        # type: () -> torch.tensor
        return self._bb_init

    def forward(self, embed):
        # type: (dt.DataItem) -> torch.tensor
        # embed size should be (# bbs, batch size, hidden size)
        _, final_state_packed = self.bb_rnn(embed, self.get_bb_init())
        final_state = final_state_packed[0]

        return self.linear(final_state.squeeze()).squeeze()
