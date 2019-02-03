import json
import logging
import os
import random
import statistics as st

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
from acoustics.generator import noise

from ..dataloader.audio import AudioDataset

logger = logging.getLogger(__name__)

#######################################################################################################################

WINDOWS = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class AudioSpectrogram(object):
    """
    Apply short-time Fourier transform (STFT)
    """

    def __init__(self, sample_rate=16000, window_size=0.02, window_stride=0.01, window='hamming'):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = WINDOWS.get(window, WINDOWS['hamming'])
        self.window_name = window

        self.n_fft = int(self.sample_rate * self.window_size)
        self.win_length = self.n_fft
        self.hop_length = int(self.sample_rate * self.window_stride)

    def __call__(self, y):
        # STFT
        d = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                         window=self.window)
        spect, _ = librosa.magphase(d)

        # S = log(S+1)
        spect = np.log1p(spect)

        return spect

    def __repr__(self):
        return self.__class__.__name__ + '(sample_rate={0}, window_size={1}, window_stride={2}, ' \
                                         'window={3})'.format(
            self.sample_rate, self.window_size, self.window_stride, self.window_name)


#######################################################################################################################


class AudioNoiseInjection(object):
    def __init__(self, probability=0.4, noise_levels=(0.0, 0.5), noise_dir='/data/noise'):
        self.probability = probability
        self.noise_levels = noise_levels
        self.noise_dir = noise_dir

        self.manifest = self.load_manifest(self.noise_dir)

    def __call__(self, data):
        if random.random() < self.probability:
            noise_entry = np.random.choice(self.manifest)
            noise_level = np.random.uniform(*self.noise_levels)
            data = self.inject_noise(data, noise_entry, noise_level)

        return data

    @staticmethod
    def inject_noise(data, noise_entry, noise_level):

        data_len = len(data)
        noise_path = noise_entry['path']
        noise_len = noise_entry['length']

        assert noise_len > data_len, "Noise file must be larger than speech sample."

        noise_start = int(np.random.rand()) * (noise_len - data_len)
        noise_stop = noise_start + data_len
        noise_data, bg_sr = sf.read(noise_path, start=noise_start, stop=noise_stop, dtype='float32')

        assert len(data) == len(noise_data), "Data and background noise length different."

        noise_energy = np.sqrt(noise_data.dot(noise_data) / noise_data.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_data * data_energy / noise_energy
        return data

    @staticmethod
    def load_manifest(noise_dir):
        path = os.path.join(noise_dir, 'manifest.json')
        with open(path, "r") as fd:
            manifest = json.load(fd)
        return manifest

    @staticmethod
    def create_manifest(noise_dir, sr, verbose=0):

        vprint = logger.info if verbose > 0 else lambda *a, **k: None

        vprint('Creating background noise manifest: ' + noise_dir)

        assert os.path.exists(noise_dir), "Missing background noise dir."

        paths = librosa.util.find_files(noise_dir)
        assert len(paths) > 0, "No background noise files found"

        def get_duration(path, sr):  # convert sampling rate and data type
            with sf.SoundFile(path) as f:
                length = len(f)
                assert f.samplerate == sr, "Background noise file have wrong sample rate."
            return length

        manifest = []
        for path in paths:
            length = get_duration(path, sr)
            manifest.append({'path': path, 'sr': sr, 'length': length})

        vprint('Total entries: ' + str(len(manifest)))

        path = os.path.join(noise_dir, 'manifest.json')
        with open(path, 'w') as outfile:
            json.dump(manifest, outfile)

        vprint('Manifest file in: ' + path)
        vprint('... complete. ')

    def __repr__(self):
        return self.__class__.__name__ + '(probability={0}, noise_levels={1}, noise_dir{2})'.format(
            self.probability, self.noise_levels, self.noise_dir)


#######################################################################################################################

# https://github.com/drscotthawley/audio-classifier-keras-cnn/blob/master/augment_data.py
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47388

class AudioPitchShift(object):
    def __init__(self, probability=0.4, sample_rate=16000, pitch_pm=4):
        self.probability = probability
        self.sample_rate = sample_rate
        self.pitch_pm = pitch_pm

    def __call__(self, data):
        if random.random() < self.probability:
            data = self.change_pitch(data, self.sample_rate, self.pitch_pm)
        return data

    @staticmethod
    def change_pitch(data, sr, pitch_pm):
        # pitch_pm - +/- this many quarter steps
        bins_per_octave = 24  # pitch increments are quarter-steps
        pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
        return librosa.effects.pitch_shift(data, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)

    def __repr__(self):
        return self.__class__.__name__ + '(probability={0}, sample_rate={1}, pitch_pm={2})'.format(
            self.probability, self.sample_rate, self.pitch_pm)


#######################################################################################################################

class AudioTimeStrech(object):
    def __init__(self, probability=0.4, low_high=(0.9, 1.1)):
        self.probability = probability
        self.low_high = low_high

        assert isinstance(self.low_high, tuple), "Parameter error - need low high tuple"

    def __call__(self, data):
        if random.random() < self.probability:
            data = self.time_stretch(data, self.low_high[0], self.low_high[1])
        return data

    @staticmethod
    def time_stretch(data, low, high):
        speed_change = np.random.uniform(low=low, high=high)
        tmp = librosa.effects.time_stretch(data, speed_change)

        length = len(data)
        minlen = min(data.shape[0], tmp.shape[0])
        return np.r_[tmp[0:minlen], np.random.uniform(-0.001, 0.001, length - minlen)]

    def __repr__(self):
        return self.__class__.__name__ + '(probability={0}, low_high={1})'.format(self.probability, self.low_high)


#######################################################################################################################

class AudioDynamicRange(object):
    def __init__(self, probability=0.4, low_high=(0.9, 1.1)):
        self.probability = probability
        self.low_high = low_high

        assert isinstance(self.low_high, tuple), "Parameter error - need low high tuple"

    def __call__(self, data):
        if random.random() < self.probability:
            data = self.dynamic_range(data, self.low_high[0], self.low_high[1])
        return data

    @staticmethod
    def dynamic_range(data, low=0.5, high=1.1):
        dyn_change = np.random.uniform(low=low, high=high)  # change amplitude
        return data * dyn_change

    def __repr__(self):
        return self.__class__.__name__ + '(probability={0}, low_high={1})'.format(self.probability, self.low_high)


#######################################################################################################################


class AudioTimeShift(object):
    def __init__(self, probability=0.4, sample_rate=16000, min_max=(-10, 10)):
        self.probability = probability
        self.sample_rate = sample_rate
        self.min_max = min_max

    def __call__(self, data):
        if random.random() < self.probability:
            shift_ms = random.uniform(self.min_max[0], self.min_max[1])
            data = self.shift_in_time(data, self.sample_rate, shift_ms)
        return data

    @staticmethod
    def shift_in_time(data, sample_rate, shift_ms):
        """
        https://github.com/PaddlePaddle/DeepSpeech/blob/develop/data_utils/audio.py
        Shift the audio in time. If `shift_ms` is positive, shift with time
        advance; if negative, shift with time delay. Silence are padded to
        keep the duration unchanged.
        """
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            data[:-shift_samples] = data[shift_samples:]
            data[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            data[-shift_samples:] = data[:shift_samples]
            data[:-shift_samples] = 0
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(probability={0}, sample_rate={1}, min_max={2})'.format(
            self.probability, self.sample_rate, self.min_max)


#
#
# class AudioTimeShift(object):
#     def __init__(self, probability=0.4, percent=0.2):
#         self.probability = probability
#         self.percent = percent
#
#     def __call__(self, data):
#         if random.random() < self.probability:
#             data = self.shift_in_time(data, self.percent)
#         return data
#
#     @staticmethod
#     def shift_in_time(data, perc):
#         length = data.shape[0]
#         timeshift_fac = perc * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
#
#         start = int(length * timeshift_fac)
#         if start >= 0:
#             data = np.r_[data[start:], np.random.uniform(-0.001, 0.001, start)]
#         else:
#             data = np.r_[np.random.uniform(-0.001, 0.001, -start), data[:start]]
#         return data
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(probability={0}, percent={1})'.format(self.probability, self.percent)


#######################################################################################################################

class AudioNoiseGeneration(object):
    def __init__(self, probability=0.4, noise_levels=(0.0, 0.5), noise_colors=['white']):
        self.probability = probability
        self.noise_levels = noise_levels
        self.noise_colors = noise_colors

    def __call__(self, data):
        if random.random() < self.probability:
            noise_color = np.random.choice(self.noise_colors)
            noise_level = np.random.uniform(*self.noise_levels)
            data = self.inject_noise(data, noise_color, noise_level)
        return data

    @staticmethod
    def inject_noise(data, noise_color, noise_level):
        noise_data = noise(len(data), color=noise_color)

        noise_energy = np.sqrt(noise_data.dot(noise_data) / noise_data.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_data * data_energy / noise_energy
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(probability={0}, noise_levels={1}, noise_colors{2})'.format(
            self.probability, self.noise_levels, self.noise_colors)


#######################################################################################################################

# https://github.com/drscotthawley/audio-classifier-keras-cnn/blob/master/augment_data.py
# https://github.com/PaddlePaddle/DeepSpeech/blob/develop/data_utils/augmentor/augmentation.py

# class AudioAugmentation(object):
#     def __init__(self, audio_conf):
#         self.audio_conf = audio_conf
#         self.sample_rate = audio_conf['sample_rate']
#         self.augment_prob = audio_conf['augment_prob']
#         self.augment_speed = audio_conf['augment_speed']
#         self.augment_dynamic = audio_conf['augment_dynamic']
#         self.augment_shift = audio_conf['augment_shift']
#         self.augment_pitch = audio_conf['augment_pitch']
#
#     # change pitch (w/o speed)
#     @staticmethod
#     def change_pitch(data, sr=16000, pitch_pm=4):
#         # pitch_pm +/- this many quarter steps
#         bins_per_octave = 24  # pitch increments are quarter-steps
#         pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
#         return librosa.effects.pitch_shift(data, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
#
#     # change speed (w/o pitch)
#     @staticmethod
#     def change_speed(data, low=0.9, high=1.1):
#         length = len(data)
#         speed_change = np.random.uniform(low=low, high=high)
#         tmp = librosa.effects.time_stretch(data, speed_change)
#
#         minlen = min(data.shape[0], tmp.shape[0])
#         return np.r_[data[0:minlen], np.random.uniform(-0.001, 0.001, length - minlen)]
#
#     # change dynamic range
#     @staticmethod
#     def change_dynamic_range(data, low=0.5, high=1.1):
#         dyn_change = np.random.uniform(low=low, high=high)  # change amplitude
#         return data * dyn_change
#
#     # shift in time forwards or backwards
#     @staticmethod
#     def shift_in_time(data, perc=0.2):
#         length = data.shape[0]
#         timeshift_fac = perc * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
#
#         start = int(length * timeshift_fac)
#         if start >= 0:
#             data = np.r_[data[start:], np.random.uniform(-0.001, 0.001, start)]
#         else:
#             data = np.r_[np.random.uniform(-0.001, 0.001, -start), data[:start]]
#         return data
#
#     def __call__(self, data):
#         if random.random() < self.augment_prob:
#             data = self.change_pitch(data, self.sample_rate, self.augment_pitch)
#             data = self.change_speed(data, *self.augment_speed)
#             data = self.change_dynamic_range(data, *self.augment_dynamic)
#             data = self.shift_in_time(data, self.augment_shift)
#         return data


#######################################################################################################################

class AudioNormalizeDB(object):
    def __init__(self, db=-20, max_gain_db=300):
        self.db = db
        self.max_gain_db = max_gain_db

    def __call__(self, data):
        """Normalize audio to be of the desired RMS value in decibels.
        """
        rms_db = self.get_rms_db(data)
        assert (self.db - rms_db) <= self.max_gain_db, "Unable to normalize - gain > max_gain_db "

        gain = min(self.max_gain_db, self.db - rms_db)
        data *= 10. ** (gain / 20.)

        return data

    @staticmethod
    def compute_rms_db(dataset, datasets=['train'], samples=None):
        manifest = list(filter(lambda entry: entry['dataset'] in datasets, dataset.manifest))

        length = samples if samples is not None else len(manifest)
        ids = list(range(0, length))
        sampled_manifest = np.array(manifest)[ids]

        rms_list = []
        for entry in sampled_manifest:
            data = AudioDataset.load_audio(entry['recording_path'], entry['recording_sr'])
            rms_list += [AudioNormalizeDB.get_rms_db(data)]

        return min(rms_list), max(rms_list), st.mean(rms_list)

    @staticmethod
    def get_rms_db(data):
        """Return root mean square energy of the audio in decibels.
        """
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(data ** 2)
        return 10 * np.log10(mean_square)

    def __repr__(self):
        return self.__class__.__name__ + '(db={0}, max_gain_db={1})'.format(self.db, self.max_gain_db)


#######################################################################################################################

class AudioNormalize(object):
    """
    Normalize spectrogram of mono audio with mean and standard deviation.
    """

    def __init__(self, _mean=None, _std=None):
        self._mean = _mean
        self._std = _std

    def __call__(self, data):
        _mean = data.mean(axis=None) if self._mean is None else self._mean
        _std = data.std(axis=None) if self._std is None else self._std
        return (data - _mean) / _std

    @staticmethod
    def compute_mean_std(dataset, transform, datasets=['train'], samples=None):
        manifest = list(filter(lambda entry: entry['dataset'] in datasets, dataset.manifest))

        length = samples if samples is not None else len(manifest)
        ids = list(range(0, length))
        sampled_manifest = np.array(manifest)[ids]

        features = []
        for entry in sampled_manifest:
            data = AudioDataset.load_audio(entry['recording_path'], entry['recording_sr'])
            spect = transform(data)
            features.append(spect)

        features = np.hstack(features)
        _mean = np.mean(features, axis=None)
        _std = np.std(features, axis=None)
        return _mean, _std

    @staticmethod
    def get_mean_std(data):
        return np.mean(data, axis=None), np.std(data, axis=None)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self._mean, self._std)


#######################################################################################################################

class AudioAugmentation(object):
    def __init__(self, transform, probability=0.9):
        self.transform = transform
        self.probability = probability

    def __call__(self, data):
        if random.random() < self.probability:
            data = self.transform(data)
        return data

    def __repr__(self):
        _repr = self.__class__.__name__ + ' [\n   '
        _repr += ',\n   '.join([repr(tr) for tr in self.transform.transforms])
        _repr += '\n], probability={0})'.format(self.probability)
        return _repr

#######################################################################################################################
