import numpy as np
import torch


#######################################################################################################################

class FromNumpyToTensor(object):
    def __init__(self, tensor_type=torch.LongTensor):
        self.tensor_type = tensor_type

    def __call__(self, arr):
        return self.tensor_type(arr)

    def __repr__(self):
        return self.__class__.__name__ + '(tensor_type={})'.format(self.tensor_type.__name__)


#######################################################################################################################

class TranscriptEncodeCTC(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, transcript):
        transcript = list(filter(None, [self.vocab(token) for token in list(str(transcript))]))
        assert len(transcript) > 0, "Empty target string."
        return np.array(transcript)

    def __repr__(self):
        return self.__class__.__name__ + '(vocab={})'.format(self.vocab)


#######################################################################################################################

class TranscriptEncodeSTS(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, transcript):
        transcript = list(filter(None, [self.vocab(token) for token in list(str(transcript))]))
        assert len(transcript) > 0, "Empty target string."

        transcript = transcript + [self.vocab('<EOS>')]
        return np.array(transcript)

    def __repr__(self):
        return self.__class__.__name__ + '(vocab={})'.format(self.vocab)

#######################################################################################################################
