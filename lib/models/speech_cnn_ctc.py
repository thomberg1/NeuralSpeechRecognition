import torch
import torch.nn as nn

from ..utilities import SequenceWise


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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


#######################################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


#######################################################################################################################

class CNN(nn.Module):
    def __init__(self, dropout=0.5, initialize=None):
        super(CNN, self).__init__()
        self.dropout = dropout
        self.initialize = initialize
        self.inplanes = 32

        self.conv1 = MaskModule(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))

        self.conv2 = MaskModule(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))

        self.bb1 = MaskModule(self._make_layer(BasicBlock, 64, 3, stride=(2, 1)))
        self.bb2 = MaskModule(self._make_layer(BasicBlock, 128, 4, stride=(2, 1)))
        self.bb3 = MaskModule(self._make_layer(BasicBlock, 256, 6, stride=(2, 1)))
        self.bb4 = MaskModule(self._make_layer(BasicBlock, 256, 3, stride=(2, 1)))

        self.conv3 = MaskModule(nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        ))

        if self.initialize is not None:
            self.initialize(self)

    def forward(self, inputs, input_sizes):
        outputs, output_sizes = self.conv1(inputs, input_sizes)

        outputs, output_sizes = self.conv2(outputs, output_sizes)

        outputs, output_sizes = self.bb1(outputs, output_sizes)
        outputs, output_sizes = self.bb2(outputs, output_sizes)
        outputs, output_sizes = self.bb3(outputs, output_sizes)
        outputs, output_sizes = self.bb4(outputs, output_sizes)

        outputs, output_sizes = self.conv3(outputs, output_sizes)

        return outputs, output_sizes

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


#######################################################################################################################

class SpeechCNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, dropout=0.5, initialize=None):
        super(SpeechCNN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.initialize = initialize

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cnn = CNN(dropout=self.dropout, initialize=initialize)

        fully_connected = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.num_classes))
        self.fc = SequenceWise(fully_connected)

    def forward(self, x, lengths, labels=None, label_sizes=None):
        x, output_lengths = self.cnn(x, lengths)

        x = x.squeeze(2).transpose(1, 2).transpose(0, 1)

        x = self.fc(x)

        x = x.transpose(0, 1)

        return x, output_lengths

#######################################################################################################################
