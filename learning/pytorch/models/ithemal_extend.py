import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import torch
import torch.nn as nn

from graph_models import AbstractGraphModule


class RNNExtend(AbstractGraphModule):

    def __init__(self, params):
        # type: (RnnParameters) -> None
        super(RNNExtend, self).__init__(params.embedding_size, params.hidden_size, params.num_classes)

        self.params = params

        # assuming LSTM for now
        self.bb_rnn = nn.LSTM(self.embedding_size, self.hidden_size)

        self._bb_init = self.rnn_init_hidden()

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def rnn_init_hidden(self):
        # type: () -> Union[Tuple[nn.Parameter, nn.Parameter], nn.Parameter]
       return self.init_hidden()

    def get_bb_init(self):
        # type: () -> torch.tensor
        return self._bb_init

    def forward(self, batch):
        # type: (dt.DataItem) -> torch.tensor
        # embed size should be (# bbs, batch size, hidden size)
        embed = torch.stack(batch.x).unsqueeze(1)
        _, final_state_packed = self.bb_rnn(embed, self.get_bb_init())
        final_state = final_state_packed[0]

        return self.linear(final_state.squeeze()).squeeze()
