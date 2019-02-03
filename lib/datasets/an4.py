import concurrent.futures
import io
import json
import logging
import multiprocessing
import os
import tarfile
from shutil import copyfile

import soundfile as sf
import torch
import torchaudio.transforms as transforms
import wget

from ..dataloader.audio import AudioDataset, BucketingSampler, collate_fn
from ..transforms.audio import AudioSpectrogram, AudioNoiseInjection, AudioNormalizeDB, AudioNormalize, \
    AudioPitchShift, AudioTimeStrech, AudioDynamicRange, AudioTimeShift, \
    AudioNoiseGeneration, AudioAugmentation
from ..transforms.general import FromNumpyToTensor, TranscriptEncodeSTS, TranscriptEncodeCTC
from ..vocabulary import Vocabulary

DEFAULT_SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


#######################################################################################################################

def create_manifest(root_path='./data/AN4', manifest_file='manifest.json',
                    url='http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz',
                    num_workers=multiprocessing.cpu_count() // 2, verbose=0):
    vprint = logger.info if verbose > 0 else lambda *a, **k: None

    vprint('Creating AN4 dataset in root dir: ' + root_path)
    extract_path = os.path.join(root_path, 'extract')
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    vprint('Tar file download: ' + url)
    tar_path = os.path.join(extract_path, url.split("/")[-1])
    download_dataset(url, tar_path, extract_path)

    vprint('Tar file xtraction: ' + tar_path + "   " + extract_path)
    extract_tarfile(tar_path, extract_path)

    datasets = [
        ('train', 'an4/etc/an4_train.fileids', 'an4/etc/an4_train.transcription'),
        ('valid', 'an4/etc/an4_test.fileids', 'an4/etc/an4_test.transcription'),
        ('test', 'an4/etc/an4_test.fileids', 'an4/etc/an4_test.transcription')
    ]

    manifest = []
    max_seq_length = 0
    for dataset, fileids_path, transcription_path in datasets:
        vprint('Processing dataset: ' + dataset + '  ' + fileids_path + '  ' + transcription_path)

        dataset_mf, seq_length = process_dataset(root_path, extract_path, fileids_path, transcription_path, dataset,
                                                 num_workers)

        manifest.extend(dataset_mf)
        if seq_length > max_seq_length:
            max_seq_length = seq_length

        vprint('Total Entries: ' + str(len(dataset_mf)))

    path = os.path.join(root_path, manifest_file)
    with open(path, 'w') as outfile:
        json.dump({'manifest': manifest, 'max_seq_length': max_seq_length}, outfile)

    vprint('AN4 creation completed - manifest file:' + path)
    vprint('Total Entries: ' + str(len(manifest)))
    vprint('... complete.')


def process_dataset(root_path, extract_path, fileids_path, transcription_path, dataset, num_workers):
    fileids_path = os.path.join(extract_path, fileids_path)
    transcription_path = os.path.join(extract_path, transcription_path)

    audio_list = process_audio(root_path, extract_path, fileids_path, dataset, num_workers)

    transcript_list, max_seq_length = process_transcripts(root_path, transcription_path, dataset, num_workers)
    transcript_dir = {os.path.basename(entry[0])[0:-4]: entry for entry in transcript_list}

    manifest = []
    for audio_entry in audio_list:
        file_name = os.path.basename(audio_entry[0])[0:-4]
        transcript_entry = transcript_dir[file_name]

        assert file_name == os.path.basename(transcript_entry[0])[0:-4], "Audio / Transcript file mismatch."

        manifest.append({'recording_path': audio_entry[0],
                         'recording_sr': audio_entry[1],
                         'recording_samples': audio_entry[2],
                         'recording_duration': audio_entry[3],
                         'transcript_path': transcript_entry[0],
                         'transcript_length': transcript_entry[1],
                         'transcript_confidence': 1.0,
                         'dataset': dataset})

    return manifest, max_seq_length


def process_audio(root_path, extract_path, fileids_path, dataset, num_workers):
    out_path = os.path.join(root_path, dataset, 'wav')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(fileids_path, 'r') as f:
        file_paths = f.read().splitlines()

    file_manifest = []
    for path in file_paths:
        raw_path = path + '.raw'
        file_manifest.append((os.path.join(extract_path, 'an4', 'wav', raw_path),
                              os.path.join(out_path, os.path.basename(path) + '.wav')))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        res = executor.map(convert_audio, file_manifest)
        audio_list = list(res)

    return audio_list


def process_transcripts(root_path, transcription_path, dataset, num_workers):
    out_path = os.path.join(root_path, dataset, 'txt')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(transcription_path, 'r') as t:
        transcripts = t.readlines()

    def clean_transcript(entry):
        entry = entry.split('(')
        text = entry[0].strip('<s>').replace('</s>', '').strip().upper()
        name = entry[1].strip().strip(')')
        return text, os.path.join(out_path, name + '.txt')

    transcript_manifest = [clean_transcript(transcript) for transcript in transcripts]
    max_seq_length = max([len(entry[0]) for entry in transcript_manifest])

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        res = executor.map(convert_transcript, transcript_manifest)
        transcript_list = list(res)

    return transcript_list, max_seq_length


def convert_audio(entry):
    raw_path, wav_path = entry[0], entry[1]

    sr, samples, duration = DEFAULT_SAMPLE_RATE, 0, 0

    with sf.SoundFile(raw_path, 'r', channels=1, samplerate=sr, subtype='PCM_16', endian='BIG') as f:
        data = f.read(frames=-1, dtype='int16')
        samples = len(f)
        sr = f.samplerate
        duration = len(f) / float(sr)

        sf.write(wav_path, data, sr, subtype='PCM_16')

        f.close()

    return wav_path, sr, samples, duration


def convert_transcript(entry):
    transcript, path = entry[0], entry[1]

    with io.FileIO(path, "w") as file:
        file.write(transcript.encode('utf-8'))

    return path, len(transcript)


def download_dataset(url, tar_path, extract_path):
    if not os.path.exists(tar_path):
        wget.download(url, out=extract_path)


def extract_tarfile(tar_path, extract_path):
    if not os.path.exists(tar_path[:-7]):
        tar = tarfile.open(tar_path)
        tar.extractall(extract_path)
        tar.close()


#######################################################################################################################


def create_pseudo_manifest(root_path, input_files, hypotheses, manifest_file='manifest_pseudo.json', verbose=1):
    vprint = logger.info if verbose > 0 else lambda *a, **k: None

    vprint('Creating pseudo dataset in root dir: ' + root_path)

    pseudo_path = os.path.join(root_path, 'pseudo')
    if not os.path.exists(pseudo_path):
        os.makedirs(pseudo_path)
        os.makedirs(os.path.join(pseudo_path, 'wav'))
        os.makedirs(os.path.join(pseudo_path, 'txt'))

    res = [(f, h) for f, h in zip(input_files, hypotheses)]

    manifest = []
    max_seq_length = 0
    for audio_path, hypothesis in res:
        name = os.path.basename(audio_path)[:-4]

        pseudo_wav_path = os.path.join(pseudo_path, 'wav', name + '.wav')
        copyfile(audio_path, pseudo_wav_path)
        samples, sr, duration = get_audio_parameters(pseudo_wav_path)

        pseudo_txt_path = os.path.join(pseudo_path, 'txt', name + '.txt')
        txt, proba = hypothesis
        writefile(pseudo_txt_path, txt)
        length = len(txt)
        if length > max_seq_length:
            max_seq_length = length

        manifest.append({'recording_path': pseudo_wav_path,
                         'recording_sr': sr,
                         'recording_samples': samples,
                         'recording_duration': duration,
                         'transcript_path': pseudo_txt_path,
                         'transcript_length': length,
                         'transcript_confidence': proba,
                         'dataset': 'pseudo'})

    path = os.path.join(root_path, manifest_file)
    with open(path, 'w') as outfile:
        json.dump({'manifest': manifest, 'max_seq_length': max_seq_length}, outfile)

    vprint('Creation completed - manifest file: ' + path)
    vprint('Total Entries: ' + str(len(manifest)))
    vprint('... done.')


def writefile(path, text):
    with io.FileIO(path, "w") as file:
        file.write(text.encode('utf-8'))


def get_audio_parameters(path):
    with sf.SoundFile(path, 'r') as f:
        samples = len(f)
        sr = f.samplerate
        duration = len(f) / float(sr)

    return samples, sr, duration


#######################################################################################################################


def create_data_pipelines(H):
    vocab = Vocabulary(os.path.join(H.ROOT_DIR, H.EXPERIMENT), encoding=H.TARGET_ENCODING)

    augmentation_transform = transforms.Compose([
        AudioNoiseInjection(probability=H.NOISE_BG_PROBABILITY,
                            noise_levels=H.NOISE_BG_LEVELS,
                            noise_dir=H.NOISE_BG_DIR),
        AudioNoiseGeneration(probability=H.AUDIO_NOISE_PROBABILITY,
                             noise_levels=H.AUDIO_NOISE_LEVELS,
                             noise_colors=H.AUDIO_NOISE_COLORS),
        AudioPitchShift(probability=H.AUDIO_PITCH_PROBABILITY,
                        sample_rate=H.AUDIO_SAMPLE_RATE,
                        pitch_pm=H.AUDIO_PITCH_PM),
        AudioTimeStrech(probability=H.AUDIO_SPEED_PROBABILITY,
                        low_high=H.AUDIO_SPEED_LOW_HIGH),
        AudioDynamicRange(probability=H.AUDIO_DYNAMIC_PROBABILITY,
                          low_high=H.AUDIO_DYNAMIC_LOW_HIGH),
        AudioTimeShift(probability=H.AUDIO_SHIFT_PROBABILITY,
                       sample_rate=H.AUDIO_SAMPLE_RATE,
                       min_max=H.AUDIO_SHIFT_MIN_MAX),
    ])

    audio_transform_train = transforms.Compose([
        AudioAugmentation(augmentation_transform, probability=H.AUGMENTATION_PROBABILITY),
        AudioNormalizeDB(db=H.NORMALIZE_DB,
                         max_gain_db=H.NORMALIZE_MAX_GAIN),
        AudioSpectrogram(sample_rate=H.AUDIO_SAMPLE_RATE,
                         window_size=H.SPECT_WINDOW_SIZE,
                         window_stride=H.SPECT_WINDOW_STRIDE,
                         window=H.SPECT_WINDOW),
        AudioNormalize(),
        FromNumpyToTensor(tensor_type=torch.FloatTensor)
    ])

    audio_transform = transforms.Compose([
        AudioNormalizeDB(db=H.NORMALIZE_DB,
                         max_gain_db=H.NORMALIZE_MAX_GAIN),
        AudioSpectrogram(sample_rate=H.AUDIO_SAMPLE_RATE,
                         window_size=H.SPECT_WINDOW_SIZE,
                         window_stride=H.SPECT_WINDOW_STRIDE,
                         window=H.SPECT_WINDOW),
        AudioNormalize(),
        FromNumpyToTensor(tensor_type=torch.FloatTensor)
    ])

    if 'ctc' in H.TARGET_ENCODING:
        label_transform = transforms.Compose([
            TranscriptEncodeCTC(vocab),
            FromNumpyToTensor(tensor_type=torch.LongTensor)
        ])
    elif 'sts' in H.TARGET_ENCODING:
        label_transform = transforms.Compose([
            TranscriptEncodeSTS(vocab),
            FromNumpyToTensor(tensor_type=torch.LongTensor)
        ])
    else:
        raise ValueError('TARGET_ENCODING value not valid.')

    train_dataset = AudioDataset(os.path.join(H.ROOT_DIR, H.EXPERIMENT),
                                 manifests_files=H.MANIFESTS, datasets=["train", "pseudo"],
                                 transform=audio_transform_train, label_transform=label_transform,
                                 max_data_size=None, sorted_by='recording_duration',
                                 min_max_duration=H.MIN_MAX_AUDIO_DURATION, min_max_length=H.MIN_MAX_TRANSCRIPT_LEN,
                                 min_confidence=H.MIN_TRANSCRIPT_CONFIDENCE)

    train_sampler = BucketingSampler(train_dataset, batch_size=H.BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=H.NUM_WORKERS, batch_sampler=train_sampler,
                                               collate_fn=collate_fn, pin_memory=True)

    logger.info(train_dataset)

    valid_dataset = AudioDataset(os.path.join(H.ROOT_DIR, H.EXPERIMENT), manifests_files=H.MANIFESTS, datasets="test",
                                 transform=audio_transform, label_transform=label_transform, max_data_size=None,
                                 sorted_by='recording_duration')

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS,
                                               shuffle=False, collate_fn=collate_fn, pin_memory=True)

    logger.info(valid_dataset)

    return train_loader, valid_loader, vocab

#######################################################################################################################
