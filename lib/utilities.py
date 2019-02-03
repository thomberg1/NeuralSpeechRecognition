import collections
import datetime
import inspect
import logging
import os
import signal
import types
from pprint import pprint

import dill
import torch
import torch.nn.init as init
from pycuda import autoinit, driver
from torch import nn


#######################################################################################################################

def gpu_stat():
    if torch.cuda.is_available():

        def pretty_bytes(byte_, precision=1):
            abbrevs = (
                (1 << 50, 'PB'), (1 << 40, 'TB'), (1 << 30, 'GB'), (1 << 20, 'MB'), (1 << 10, 'kB'), (1, 'bytes'))
            if byte_ == 1:
                return '1 byte'
            factor, suffix = 1, ''
            for factor, suffix in abbrevs:
                if byte_ >= factor:
                    break
            return '%.*f%s' % (precision, byte_ / factor, suffix)

        device = autoinit.device
        print('GPU Name: %s' % device.name())
        print('GPU Memory: %s' % pretty_bytes(device.total_memory()))
        print('CUDA Version: %s' % str(driver.get_version()))
        print('GPU Free/Total Memory: %d%%' % ((driver.mem_get_info()[0] / driver.mem_get_info()[1]) * 100))


#######################################################################################################################


class HYPERPARAMETERS(collections.OrderedDict):
    """
    Class to make it easier to access hyper parameters by either dictionary or attribute syntax.
    """

    def __init__(self, dictionary):
        super(HYPERPARAMETERS, self).__init__(dictionary)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

    def pprint(self, path):
        with open(path, "w+") as h_file:
            pprint(self, stream=h_file)

    @staticmethod
    def create_timestamp(ts=None):
        ts = datetime.datetime.now().timestamp() if ts is None else ts
        dts = datetime.datetime.fromtimestamp(ts)
        return '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{:06d}'.format(dts.year, dts.month, dts.day, dts.hour,
                                                                         dts.minute, dts.second, dts.second,
                                                                         dts.microsecond)

    @staticmethod
    def convert_timestamp(time_str=None):
        dts = datetime.datetime(*[int(parts) for parts in time_str.split('-')])
        return dts.timestamp()

    @staticmethod
    def load(path):
        with open(path, 'rb') as in_strm:
            h = dill.load(in_strm)
        return h

    @staticmethod
    def dump(h, path):
        with open(path, 'wb') as out_strm:
            dill.dump(h, out_strm)

    def __repr__(self):
        fmt_str = '{' + '\n'
        for k, v in self.items():
            if '__class__' in k:
                continue
            if isinstance(v, types.LambdaType):  # function or lambda
                if v.__name__ in '<lambda>':
                    try:
                        fmt_str += inspect.getsource(v)
                    except:
                        fmt_str += "    " + "'{}'".format(k).ljust(32) + ": '" + str(v) + "' ,\n"
                else:
                    fmt_str += "    " + "'{}'".format(k).ljust(32) + ': ' + v.__name__ + ' ,\n'
            elif isinstance(v, type):  # class
                fmt_str += "    " + "'{}'".format(k).ljust(32) + ': ' + v.__name__ + ' ,\n'
            else:  # everything else
                if isinstance(v, str):
                    fmt_str += "    " + "'{}'".format(k).ljust(32) + ": '" + str(v) + "' ,\n"
                else:
                    fmt_str += "    " + "'{}'".format(k).ljust(32) + ': ' + str(v) + ' ,\n'
        fmt_str += '}\n'
        return fmt_str


#######################################################################################################################


class Metric(object):
    """
    Class to track runtime statistics easier. Inspired by History Variables that not only store the current value,
    but also the values previously assigned. (see https://rosettacode.org/wiki/History_variables)
    """

    def __init__(self, metrics):
        self.metrics = [m[0] for m in metrics]
        self.init_vals = {m[0]: m[1] for m in metrics}
        self.values = {}
        for name in self.metrics:
            self.values[name] = []

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name in self.metrics:
            self.values[name].append(value)

    def __getattr__(self, attr):
        if attr in self.metrics and not len(self.values[attr]):
            val = self.init_vals[attr]
        else:
            val = self.__dict__[attr]
        return val

    def values(self, metric):
        return self.values[metric]

    def state_dict(self):
        state = {}
        for m in self.metrics:
            state[m] = self.values[m]
        return state

    def load_state_dict(self, state_dict):
        for m in state_dict:
            self.values[m] = state_dict[m]


#######################################################################################################################
# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5


def torch_weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)


#######################################################################################################################

def create_logger(H):
    if not os.path.exists(H.EXPERIMENT):
        os.makedirs(H.EXPERIMENT)

    logFormatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')

    fileHandler = logging.FileHandler("{0}/{1}.log".format(H.EXPERIMENT, H.MODEL_NAME))
    fileHandler.setFormatter(logFormatter)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)

    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                        level=logging.INFO, handlers=[consoleHandler, fileHandler])


#######################################################################################################################
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py


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
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


#######################################################################################################################
# https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py


class DelayedKeyboardInterrupt(object):
    def __init__(self):
        self.signal_received = None

    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type_, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

#######################################################################################################################
