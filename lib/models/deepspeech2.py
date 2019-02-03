import math
from collections import OrderedDict

import torch
import torch.nn as nn


#######################################################################################################################

class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.ByteTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


#######################################################################################################################

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


#######################################################################################################################


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=True, dropout=0.5):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None

        self.drop_out = SequenceWise(nn.Dropout(self.dropout))

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                          bias=False if batch_norm else True)

        self.num_directions = 2 if bidirectional else 1

    def forward(self, x, output_lengths):

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)

        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum

        x = self.drop_out(x)

        return x


#######################################################################################################################

class CNN(nn.Module):
    def __init__(self, dropout=0.5, initialize=None):
        super(CNN, self).__init__()
        self.initialize = initialize

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout)
        ))

        if self.initialize is not None:
            self.initialize(self)

    def forward(self, x, lengths):

        output_lengths = self.get_seq_lens(lengths)

        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length.cpu().int()
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()


#######################################################################################################################

class DeepSpeech2(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=768, nb_layers=5,
                 bidirectional=True, cnn_dropout=0.5, rnn_dropout=0.5,
                 sample_rate=16000, window_size=0.02, initialize=None):
        super(DeepSpeech2, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = rnn_hidden_size
        self.hidden_layers = nb_layers
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.cnn_dropout = cnn_dropout
        self.rnn_dropout = rnn_dropout
        self.initialize = initialize

        # -------------------
        self.conv = CNN(dropout=self.cnn_dropout, initialize=self.initialize)

        # -------------------
        rnn_input_size = self.get_rnn_input_size(self.sample_rate, self.window_size)

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                       bidirectional=bidirectional, batch_norm=False, dropout=rnn_dropout)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size,
                           bidirectional=bidirectional, dropout=rnn_dropout)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        # ------------------
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, self.num_classes, bias=False)
        )
        self.fc = SequenceWise(fully_connected)

        if self.initialize is not None:
            self.initialize(self)

    def forward(self, x, lengths, labels=None, label_sizes=None):

        x = x.cuda() if next(self.parameters()).is_cuda else x

        x, output_lengths = self.conv(x, lengths)

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        x = self.fc(x)
        x = x.transpose(0, 1)

        return x, output_lengths

    @staticmethod
    def get_rnn_input_size(sample_rate, window_size):
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1

        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        return rnn_input_size

#######################################################################################################################
