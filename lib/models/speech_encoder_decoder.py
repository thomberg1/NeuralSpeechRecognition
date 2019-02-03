import math
import random

import numpy as np
import torch
import torch.nn as nn


#######################################################################################################################


class MaskModule(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        """
        super(MaskModule, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, x_lengths):
        """
        Input of size BxCxDxT
        """
        lengths = self.get_seq_lens(x_lengths)
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

    def get_seq_lens(self, input_lengths):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        """
        seq_len = input_lengths.cpu().int()
        for m in self.seq_module.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
            elif type(m) == nn.modules.pooling.MaxPool2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
            elif type(m) == torch.nn.modules.pooling.AvgPool2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.kernel_size[1]) / m.stride[1] + 1)

        return seq_len.int()


#######################################################################################################################

class CNN(nn.Module):
    def __init__(self, dropout=0.5, initialize=None):
        super(CNN, self).__init__()
        self.initialize = initialize

        self.conv1 = MaskModule(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout)
        ))

        self.conv2 = MaskModule(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout)
        ))

        if self.initialize is not None:
            self.initialize(self)

    def forward(self, inputs, input_sizes):
        outputs, output_sizes = self.conv1(inputs, input_sizes)

        outputs, output_sizes = self.conv2(outputs, output_sizes)

        b, c, f, t = outputs.size()
        outputs = outputs.view(b, c * f, t).transpose(1, 2).contiguous()

        # B x T x F
        return outputs, output_sizes

    @staticmethod
    def get_output_size(sample_rate, window_size):
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1

        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        input_size *= 32

        return input_size


#######################################################################################################################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.5):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bias=True,
                          dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, inputs, lengths):
        pack_seq = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
        pack_seq, hidden = self.rnn(pack_seq)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(pack_seq, batch_first=True)
        return outputs, lengths, hidden


#######################################################################################################################

class Encoder(nn.Module):
    def __init__(self, cnn, input_size, hidden_size=128, num_layers=1, dropout=0.5, bidirectional=True,
                 initialize=None):
        super(Encoder, self).__init__()
        self.conv = cnn
        self.rnn_input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.initialize = initialize

        self.rnn = RNN(self.rnn_input_size, self.hidden_size, self.num_layers, dropout=self.dropout,
                       bidirectional=self.bidirectional)

        if self.initialize is not None:
            self.initialize(self)

    def forward(self, inputs, input_sizes):

        outputs, output_lengths = self.conv(inputs, input_sizes)

        outputs, output_lengths, hidden = self.rnn(outputs, output_lengths)

        hidden = self._cat(hidden)

        return outputs, output_lengths, hidden

    def _cat(self, h):
        """
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


#######################################################################################################################

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, outputs, context, mask=None):
        batch_size = outputs.size(0)
        hidden_size = outputs.size(2)
        input_size = context.size(1)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(outputs, context.transpose(1, 2))

        if mask is not None:
            attn.masked_fill_(mask, -float('inf'))

        attn = torch.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, outputs), dim=2)

        # output -> (batch, out_len, dim)
        outputs = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return outputs, attn

#######################################################################################################################

class Decoder(nn.Module):
    def __init__(self, vocab, max_seq_length, hidden_size=256, num_layers=1, dropout=0.5, teacher_forcing_ratio=0.5,
                 initialize=None):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.vocab_size = len(self.vocab)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.initialize = initialize

        self.embedding = nn.Sequential(
            nn.Embedding(self.vocab_size, self.hidden_size, sparse=False, padding_idx=0),
            nn.Dropout(0.5)
        )

        self.attn = Attention(self.hidden_size)

        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bias=True,
                          dropout=self.dropout)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        if self.initialize is not None:
            self.initialize(self)

        fix_embedding = torch.from_numpy(np.eye(self.vocab_size, self.vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, enc_inputs, enc_input_sizes, hidden, labels=None, label_sizes=None):

        if self.training:
            assert labels is not None and label_sizes is not None, "Need labels in trainings mode."

        use_cuda = next(self.parameters()).is_cuda

        batch_size = enc_inputs.size(0)
        inputs = torch.LongTensor([self.vocab("<SOS>")] * batch_size).view(batch_size, 1)
        inputs = inputs.cuda() if use_cuda else inputs

        max_length = labels.size(1) if labels is not None else self.max_seq_length + 1

        # mask = self.get_mask(enc_input_sizes).unsqueeze(1)
        # mask = mask.cuda() if use_cuda else mask
        mask = None

        dec_output_sizes = torch.LongTensor(batch_size).fill_(max_length)
        dec_output_sizes = dec_output_sizes.cuda() if use_cuda else dec_output_sizes

        dec_outputs = []
        for t in range(max_length):

            outputs, hidden = self.step(inputs, hidden, enc_inputs, mask)
            dec_outputs.append(outputs)

            inputs = outputs.topk(1)[1].view(batch_size, 1)

            dec_output_sizes[inputs.squeeze(1).eq(self.vocab('<EOS>')) * dec_output_sizes.gt(t)] = t
            if labels is None and dec_output_sizes.le(t + 1).all():
                break

            if self.training and random.random() < self.teacher_forcing_ratio:
                inputs = labels[:, t].view(batch_size, 1)

        dec_outputs = torch.cat(dec_outputs, dim=1)

        return dec_outputs, dec_output_sizes

    def step(self, inputs, hidden, enc_inputs, mask):
        batch_size, output_size = inputs.size(0), inputs.size(1)

        embeddings = self.embedding(inputs)

        outputs, hidden = self.rnn(embeddings, hidden)

        outputs, attn = self.attn(outputs, enc_inputs, mask)

        outputs = self.fc(outputs.contiguous().squeeze(1))

        outputs = torch.log_softmax(outputs, dim=1).view(batch_size, output_size, -1)

        return outputs, hidden

    @staticmethod
    def get_mask(lengths):
        batch_size = lengths.numel()
        mask = (torch.arange(0, lengths.max()).type_as(lengths).repeat(batch_size, 1).gt(lengths.unsqueeze(1)))
        return mask


#######################################################################################################################

class NeuralSpeechRecognizer(nn.Module):
    def __init__(self, vocab, max_seq_length, rnn_hidden_size=256, rnn_num_layers=1, rnn_dropout=0.5, cnn_dropout=0.5,
                 teacher_forcing_ratio=0.5, sample_rate=16000, window_size=0.02, initialize=None):
        super(NeuralSpeechRecognizer, self).__init__()
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.cnn_dropout = cnn_dropout
        self.rnn_dropout = rnn_dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.initialize = initialize

        self.cnn = CNN(dropout=self.cnn_dropout,
                       initialize=self.initialize)

        self.enc = Encoder(self.cnn,
                           input_size=self.cnn.get_output_size(self.sample_rate, self.window_size),
                           hidden_size=self.rnn_hidden_size,
                           num_layers=self.rnn_num_layers,
                           dropout=self.rnn_dropout,
                           bidirectional=True,
                           initialize=self.initialize)

        self.dec = Decoder(vocab=self.vocab,
                           max_seq_length=self.max_seq_length,
                           hidden_size=self.rnn_hidden_size * 2,
                           num_layers=self.rnn_num_layers,
                           dropout=self.rnn_dropout,
                           teacher_forcing_ratio=self.teacher_forcing_ratio,
                           initialize=self.initialize)

    def forward(self, inputs, input_sizes, labels=None, label_sizes=None):
        outputs, output_sizes, hidden = self.enc(inputs, input_sizes)

        outputs, output_sizes = self.dec(outputs, output_sizes, hidden, labels, label_sizes)

        return outputs, output_sizes

#######################################################################################################################
