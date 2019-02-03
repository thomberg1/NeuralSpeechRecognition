import json
import os
import random

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


#######################################################################################################################


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, manifests_files='manifest.json', datasets='train', transform=None,
                 label_transform=None, max_data_size=None, sorted_by=None,
                 min_max_duration=None, min_max_length=None, min_confidence=None):
        self.root_dir = root_dir
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.manifests_files = manifests_files if isinstance(manifests_files, list) else [manifests_files]
        self.transform = transform
        self.label_transform = label_transform
        self.max_data_size = max_data_size
        self.sorted_by = sorted_by  # "recording_duration" "transcript_length" None

        assert set(self.datasets).issubset({'train', 'valid', 'test', 'pseudo'}), 'dataset parameter invalid.'
        assert self.sorted_by in ['recording_duration', 'transcript_length', None], "sorted_by parameter invalid."

        # "seq" in self.encoding and self.vocab("<SOS>" != 0), "Vocabulary doesn't contain <SOS>."

        self.manifest, self.max_seq_length = self.load_manifests(self.root_dir, self.manifests_files, self.datasets,
                                                                 min_max_duration, min_max_length, min_confidence)

        if isinstance(self.max_data_size, int):
            assert self.max_data_size < len(self.manifest)  # max_train_size needs to select a subset
            self.manifest = random.sample(self.manifest, self.max_data_size)

        if self.sorted_by is not None:
            assert self.sorted_by in ['recording_duration', 'transcript_length']
            self.manifest = sorted(self.manifest, key=lambda k: k[self.sorted_by])

    def __len__(self):
        return len(self.manifest) if self.max_data_size is None else self.max_data_size

    def __getitem__(self, idx):
        annotation = self.manifest[idx]

        data = self.load_audio(annotation['recording_path'], annotation['recording_sr'])
        if self.transform is not None:
            data = self.transform(data)

        transcript = self.load_transcript(annotation['transcript_path'])
        if self.label_transform is not None:
            transcript = self.label_transform(transcript)

        return data, transcript, idx

    @staticmethod
    def load_audio(path, sr):
        from scipy.io import wavfile
        fs, data = wavfile.read(path)

        assert fs == sr, "File sampling rate doesn't match manifest."
        assert data.dtype.name == 'int16', "Required by normalization"
        assert len(data.shape) == 1, "multiple channels wav files not supported"

        data = np.asarray(data, dtype='float32') / np.iinfo(np.dtype('int16')).min  # normalization

        return data

    @staticmethod
    def load_transcript(path):
        with open(path, 'r', encoding='utf8') as f:
            transcript = f.read().strip()
        return np.array(transcript)

    @staticmethod
    def load_manifests(root, manifest_files, datasets, min_max_duration, min_max_length, min_confidence):

        # merge manifests into one
        manifest = []
        max_seq_length = 0
        for file in manifest_files:
            path = os.path.join(root, file)
            with open(path, "r") as fd:
                content = json.load(fd)
                assert len(content), 'Dataset empty.'
                manifest.extend(content['manifest'])
                if content['max_seq_length'] > max_seq_length:
                    max_seq_length = content['max_seq_length']

        # prune test set to min/max duration
        flt = filter(lambda entry: entry['dataset'] in datasets, manifest)

        if min_max_duration is not None:
            min_duration = min_max_duration[0]
            max_duration = min_max_duration[1]
            flt = filter(lambda entry: min_duration <= entry['recording_duration'] <= max_duration, flt)

        if min_max_length is not None:
            min_length = min_max_length[0]
            max_length = min_max_length[1]
            flt = filter(lambda entry: min_length <= entry['transcript_length'] <= max_length, flt)

        if min_confidence is not None:
            flt = filter(lambda entry: min_confidence <= entry['transcript_confidence'], flt)

        manifest = list(flt)

        assert len(manifest) > 0, 'No data in dataset.'

        if len(manifest_files) >= 2:
            random.shuffle(manifest)

        return manifest, max_seq_length

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Total of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Total of duration (min): {}\n'.format(
            sum([a['recording_duration'] for a in self.manifest]) / 60)
        fmt_str += '    Root Location: {}\n'.format(self.root_dir)
        hdr = '    Transforms: '
        bdy = ('\n' + ' ' * 8).join(
            [repr(tx) for tx in self.transform.transforms]) if self.transform is not None else "[]"
        fmt_str += '{0}\n        {1}\n'.format(hdr, bdy)
        hdr = '    Label Transforms: '
        bdy = ('\n' + ' ' * 8).join(
            [repr(tx) for tx in self.label_transform.transforms]) if self.label_transform is not None else "[]"
        fmt_str += '{0}\n        {1}'.format(hdr, bdy)
        return fmt_str


#######################################################################################################################

class BucketingSampler(Sampler):
    """
    Creates minibatches of siize batch_size. Samples batches assuming data is in order
    of size to batch similarly sized samples together. Shifts the samples by a random factor to
    move samples from one batch into the next.
    """

    def __init__(self, data_source, batch_size=1):
        super(BucketingSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source

        self.bins = self.create_bins(shift=False)

    def __iter__(self):
        for _bin in self.bins:
            np.random.shuffle(_bin)
            yield _bin

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        self.bins = self.create_bins(shift=True)
        np.random.shuffle(self.bins)

    def create_bins(self, shift=True):
        # shift factor for samples - start at 2 to avoid buckets of size 1
        shift_len = random.randint(2, self.batch_size - 1) if shift else 0

        ids = list(range(0, len(self.data_source)))
        bins = [ids[i:i + self.batch_size] for i in range(shift_len, len(ids), self.batch_size)]

        out_bins = [ids[0:shift_len]] if shift else []
        out_bins += bins[0: None if len(bins[-1]) > 1 else - 1]  # remove buckets of size 1 at the end
        return out_bins


#######################################################################################################################


def collate_fn(batch):
    minibatch_size = len(batch)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    freq_size, max_time_size = batch[0][0].size()

    max_label = max(batch, key=lambda sample: sample[1].size(0))
    max_label_size = max_label[1].size(0)

    inputs = torch.zeros(minibatch_size, 1, freq_size, max_time_size).float()
    input_size = torch.LongTensor(minibatch_size)
    labels = torch.zeros(minibatch_size, max_label_size).long()
    label_sizes = torch.LongTensor(minibatch_size)

    idxs = []
    for i, (input_, label, idx) in enumerate(batch):
        length = input_.size(1)
        inputs[i][0].narrow(1, 0, length).copy_(input_)
        input_size[i] = length

        length = len(label)
        labels[i].narrow(0, 0, length).copy_(label)
        label_sizes[i] = length

        idxs.append(idx)

    return inputs, labels, input_size, label_sizes, idxs


#######################################################################################################################


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, sample_rate, vocab, transform=None):
        self.audio_files = audio_files
        self.vocab = vocab
        self.transform = transform

        self.sr = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]

        data = self.load_audio(audio_file, self.sr)

        if self.transform is not None:
            data = self.transform(data)

        return data, idx

    @staticmethod
    def load_audio(path, sr):
        from scipy.io import wavfile
        fs, data = wavfile.read(path)

        assert fs == sr, "File sampling rate doesn't match manifest."
        assert data.dtype.name == 'int16', "Required by normalization"
        assert len(data.shape) == 1, "multiple channels wav files not supported"

        data = np.asarray(data, dtype='float32') / np.iinfo(np.dtype('int16')).min  # normalization

        return data


#######################################################################################################################


def dummy_collate_fn(batch):
    minibatch_size = len(batch)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    freq_size, max_time_size = batch[0][0].size()

    inputs = torch.zeros(minibatch_size, 1, freq_size, max_time_size).float()
    input_size = torch.LongTensor(minibatch_size)

    idxs = []
    for i, (tensor, idx) in enumerate(batch):
        length = tensor.size(1)
        inputs[i][0].narrow(1, 0, length).copy_(tensor)
        input_size[i] = length

        idxs.append(idx)

    return inputs, input_size, idxs

#######################################################################################################################
