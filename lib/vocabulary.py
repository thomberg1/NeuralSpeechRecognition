import json
import logging
import os

logger = logging.getLogger(__name__)


#######################################################################################################################

class Vocabulary(object):
    def __init__(self, root_dir, encoding="sts", verbose=0):
        self.root_dir = root_dir
        self.encoding = encoding
        self.verbose = verbose
        self.vprint = logger.info if self.verbose > 0 else lambda *a, **k: None

        assert self.encoding in ['sts', 'ctc'], "encoding parameter invalid."

        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.path = os.path.join(self.root_dir, 'vocab.json')

        if os.path.exists(self.path):
            self.load()
        else:
            self.vprint("Warning - empty vocabulary object.")

    def add(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def __call__(self, val):
        if isinstance(val, str):
            res = self.token2idx[val] if val in self.token2idx else None
        elif isinstance(val, int):
            res = self.idx2token[val] if val <= self.__len__() else None
        else:
            raise RuntimeError
        return res

    def __len__(self):
        return len(self.token2idx)

    def create(self, alphabet):
        for c in alphabet:
            self.add(c)

    def dump(self):
        data = {'idx': self.idx, 'token2idx': self.token2idx, 'idx2token': self.idx2token}
        with open(self.path, "w+") as fd:
            json.dump(data, fd)

        self.vprint('Created vocabular: ' + self.path)

    def load(self):
        with open(self.path, "r") as fd:
            data = json.load(fd)
            self.idx = int(data['idx'])
            self.token2idx = data['token2idx']
            self.idx2token = {int(k): v for k, v in data['idx2token'].items()}

            if 'sts' in self.encoding:
                self.add('<SOS>')
                self.add('<EOS>')
                self.add('<UNK>')
            elif 'ctc' not in self.encoding:
                raise ValueError('Parameter encoding not valid.')

    def __repr__(self):
        return ''.join(list(self.token2idx.keys()))

#######################################################################################################################
