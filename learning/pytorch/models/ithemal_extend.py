import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))
from enum import Enum, unique

import torch
import torch.nn as nn

import models.graph_models as md
from graph_models import AbstractGraphModule


class RNNExtend(AbstractGraphModule):

    def __init__(self, params, args):
        # type: (RnnParameters) -> None
        super(RNNExtend, self).__init__(params.embedding_size, params.hidden_size, params.num_classes)

        self.params = params
        self.args = args

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
        pred = self.linear(final_state.squeeze()).squeeze()
        if self.args.use_scaling:
            pred = pred * self.args.scale_amount

        return pred


class GraphNN(AbstractGraphModule):

    def __init__(self, embedding_size, hidden_size, num_classes, use_residual=True, use_dag_rnn=True, reduction=md.ReductionType.MAX, nonlinear_width=128, nonlinear_type=md.NonlinearityType.RELU, nonlinear_before_max=False):
        # type: (int, int, int, bool, bool, bool, ReductionType, int, NonlinearityType, bool) -> None
        super(GraphNN, self).__init__(embedding_size, hidden_size, num_classes)

        assert use_residual or use_dag_rnn, 'Must use some type of predictor'

        self.use_residual = use_residual
        self.use_dag_rnn = use_dag_rnn

        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

        # linear weight for instruction embedding
        self.opcode_lin = nn.Linear(self.embedding_size, self.hidden_size)
        self.src_lin = nn.Linear(self.embedding_size, self.hidden_size)
        self.dst_lin = nn.Linear(self.embedding_size, self.hidden_size)
        # for sequential model
        self.opcode_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)
        self.src_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)
        self.dst_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)

        #linear layer for final regression result
        self.linear = nn.Linear(self.hidden_size,self.num_classes)

        self.nonlinear_1 = nn.Linear(self.hidden_size, nonlinear_width)
        self.nonlinear_2 = nn.Linear(nonlinear_width, self.num_classes)

        #lstm - for sequential model
        self.lstm_token_seq = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins_seq = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear_seq = nn.Linear(self.hidden_size, self.num_classes)

        self.reduction_typ = reduction
        self.attention_1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.attention_2 = nn.Linear(self.hidden_size // 2, 1)

        self.nonlinear_premax_1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.nonlinear_premax_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.nonlinear_seq_1 = nn.Linear(self.hidden_size, nonlinear_width)
        self.nonlinear_seq_2 = nn.Linear(nonlinear_width, self.num_classes)

        self.use_nonlinear = nonlinear_type is not None

        if nonlinear_type == md.NonlinearityType.RELU:
            self.final_nonlinearity = torch.relu
        elif nonlinear_type == md.NonlinearityType.SIGMOID:
            self.final_nonlinearity = torch.sigmoid
        elif nonlinear_type == md.NonlinearityType.TANH:
            self.final_nonlinearity = torch.tanh

        self.nonlinear_before_max = nonlinear_before_max

    def reduction(self, items):
        # type: (List[torch.tensor]) -> torch.tensor
        if len(items) == 0:
            return self.init_hidden()[0]
        elif len(items) == 1:
            return items[0]

        def binary_reduction(reduction):
            # type: (Callable[[torch.tensor, torch.tensor], torch.tensor]) -> torch.tensor
            final = items[0]
            for item in items[1:]:
                final = reduction(final, item)
            return final

        stacked_items = torch.stack(items)

        if self.reduction_typ == md.ReductionType.MAX:
            return binary_reduction(torch.max)
        elif self.reduction_typ == md.ReductionType.ADD:
            return binary_reduction(torch.add)
        elif self.reduction_typ == md.ReductionType.MEAN:
            return binary_reduction(torch.add) / len(items)
        elif self.reduction_typ == md.ReductionType.ATTENTION:
            preds = torch.stack([self.attention_2(torch.relu(self.attention_1(item))) for item in items])
            probs = F.softmax(preds, dim=0)
            print('{}, {}, {}'.format(
                probs.shape,
                stacked_items.shape,
                stacked_items * probs
            ))
            return (stacked_items * probs).sum(dim=0)
        else:
            raise ValueError()

    def remove_refs(self, item):
        # type: (dt.DataItem) -> None
        for bblock in item.function.bblocks:
            if bblock.lstm != None:
                del bblock.lstm
            if bblock.hidden != None:
                del bblock.hidden
            bblock.lstm = None
            bblock.hidden = None

    def init_funclstm(self, item):
        # type: (dt.DataItem) -> None
        self.remove_refs(item)

    def create_graphlstm(self, function):
        # type: (ut.BasicBlock) -> torch.tensor
        leaves = function.find_leaves()

        leaf_hidden = []
        for leaf in leaves:
            hidden = self.create_graphlstm_rec(leaf)
            leaf_hidden.append(hidden[0].squeeze())

        if self.nonlinear_before_max:
            leaf_hidden = [
                self.nonlinear_premax_2(torch.relu(self.nonlinear_premax_1(h)))
                for h in leaf_hidden
            ]

        return self.reduction(leaf_hidden)

    def create_graphlstm_rec(self, bblock):
        # type: (ut.Instruction) -> torch.tensor
        if bblock.hidden != None:
            return bblock.hidden

        parent_hidden = [self.create_graphlstm_rec(parent) for parent in bblock.parents]

        if len(parent_hidden) > 0:
            hs, cs = list(zip(*parent_hidden))
            in_hidden_ins = (self.reduction(hs), self.reduction(cs))
        else:
            in_hidden_ins = self.init_hidden()

        if bblock.embed is None:
            hidden_ins = in_hidden_ins
        else:
            out_ins, hidden_ins = self.lstm_ins(
                bblock.embed.unsqueeze(0).unsqueeze(0), in_hidden_ins)
        bblock.hidden = hidden_ins

        return bblock.hidden

    def create_residual_lstm(self, function):
        # type: (ut.BasicBlock) -> torch.tensor
        ins_embeds_lstm = function.get_embedding().unsqueeze(1)

        _, hidden_ins = self.lstm_ins_seq(ins_embeds_lstm, self.init_hidden())

        seq_ret = hidden_ins[0].squeeze()

        return seq_ret

    def forward(self, item):
        # type: (dt.DataItem) -> torch.tensor
        self.init_funclstm(item)

        final_pred = torch.zeros(self.num_classes).squeeze()

        if self.use_dag_rnn:
            graph = self.create_graphlstm(item.function)
            if self.use_nonlinear and not self.nonlinear_before_max:
                final_pred += self.nonlinear_2(self.final_nonlinearity(self.nonlinear_1(graph))).squeeze()
            else:
                final_pred += self.linear(graph).squeeze()

        if self.use_residual:
            sequential = self.create_residual_lstm(item.function)
            if self.use_nonlinear:
                final_pred += self.nonlinear_seq_2(self.final_nonlinearity(self.nonlinear_seq_1(sequential))).squeeze()
            else:
                final_pred += self.linear(sequential).squeeze()

        return final_pred.squeeze()


class BasicBlock:

    def __init__(self, embed, name='', bb_id=None):
        self.embed = embed
        self.name = name
        self.bb_id = bb_id

        self.parents = []
        self.children = []
        self.parents_probs = []
        self.children_probs = []

        #for lstms
        self.lstm = None
        self.hidden = None
        self.tokens = None

    def print_bb(self):
        print('#####')
        print('Function: %s, BasicBlock %d' % (self.name, self.bb_id))
        print('')
        if len(self.parents) > 0:
            print('Parents:')
            for i, parent in enumerate(self.parents):
                print(parent.__str__() + ', Edge Prob = %.2f' % (self.parents_probs[i]))
            print('')
        if len(self.children) > 0:
            print('Children:')
            for i, child in enumerate(self.children):
                print(child.__str__() + ', Edge Prob = %.2f' % (self.children_probs[i]))
            print('')
        print('#####')

    def __str__(self):
        return 'Function: %s, BasicBlock %d: %d parents and %d children' % (
            self.name, self.bb_id, len(self.parents), len(self.children))


class Function:

    def __init__(self, bblocks, name=''):
        self.bblocks = bblocks
        self.name = name

    def num_bblocks(self):
        return len(self.bblocks)

    def print_function(self):
        for bblock in self.bblocks:
            bblock.print_bb()

    def get_embedding(self):
        return torch.stack(
            [bb.embed for bb in self.bblocks if bb.embed is not None])

    def linearize_edges(self):
        for fst, snd in zip(self.bblocks, self.bblocks[1:]):
            if snd not in fst.children:
                fst.children.append(snd)
            if fst not in snd.parents:
                snd.parents.append(fst)

    def find_roots(self):
        roots = []
        for bblock in self.bblocks:
            if len(bblock.parents) == 0:
                roots.append(bblock)
        return roots

    def find_leaves(self):
        leaves = []
        for bblock in self.bblocks:
            if len(bblock.children) == 0:
                leaves.append(bblock)
        return leaves
