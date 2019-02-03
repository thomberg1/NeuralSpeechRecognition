import concurrent.futures
import csv
import io
import json
import os
import subprocess
import tarfile
from multiprocessing import cpu_count

import soundfile as sf
import wget

DEFAULT_SAMPLE_RATE = 16000


#######################################################################################################################

def create_manifest(root_path='./data/CommonVoice', manifest_file='manifest.json',
                    url="https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz",
                    verbose=0):
    vprint = print if verbose > 0 else lambda *a, **k: None

    vprint('Creating CommonVoice dataset in root dir: ', root_path, end='')
    extract_path = os.path.join(root_path, 'extract')
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    vprint('... done.')

    vprint('Tar file download:', url, end='')
    tar_path = os.path.join(extract_path, url.split("/")[-1])
    download_dataset(url, tar_path, extract_path)
    vprint('... done.')

    vprint('Tar file xtraction:', tar_path, extract_path, end='')
    extract_tarfile(tar_path, extract_path)
    print('... done.')

    datasets = [
        ('train', 'cv-valid-train.csv'),
        ('valid', 'cv-valid-dev.csv'),
        ('test', 'cv-valid-test.csv')
    ]

    manifest = []
    max_seq_length = 0
    for dataset, cv_path in datasets:
        vprint('Processing dataset:', dataset, cv_path)

        cv_path = os.path.join(extract_path, 'cv_corpus_v1', cv_path)
        dataset_mf, seq_length = process_dataset(root_path, extract_path, cv_path, dataset)

        manifest.extend(dataset_mf)
        if seq_length > max_seq_length:
            max_seq_length = seq_length

        vprint('Total Entries: ', len(dataset_mf))

    path = os.path.join(root_path, manifest_file)
    with open(path, 'w') as outfile:
        json.dump({'manifest': manifest, 'max_seq_length': max_seq_length}, outfile)

    vprint('CommonVoice creation completed - manifest file:', path)
    vprint('Total Entries: ', len(manifest))
    vprint('... complete.')


def process_dataset(root_path, extract_path, cv_path, dataset):
    with open(cv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        cv_data = [(row['filename'], row['text']) for row in reader]

    assert len(cv_data), "CV file empty"

    audio_list = process_audio(root_path, extract_path, cv_data, dataset)

    transcript_list, max_seq_length = process_transcripts(root_path, cv_data, dataset)
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


def process_audio(root_path, extract_path, cv_data, dataset):
    out_path = os.path.join(root_path, dataset, 'wav')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_manifest = []
    for row in cv_data:
        path = row[0]
        file_manifest.append((os.path.join(extract_path, 'cv_corpus_v1', path),
                              os.path.join(out_path, os.path.basename(path)[:-4] + '.wav')))

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        res = executor.map(convert_audio, file_manifest)
        audio_list = list(res)

    return audio_list


def process_transcripts(root_path, cv_data, dataset):
    out_path = os.path.join(root_path, dataset, 'txt')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    def clean_transcript(row):
        name = os.path.basename(row[0])[:-4]
        text = row[1].strip().upper()
        return text, os.path.join(out_path, name + '.txt')

    transcript_manifest = [clean_transcript(row) for row in cv_data]
    max_seq_length = max([len(entry[0]) for entry in transcript_manifest])

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        res = executor.map(convert_transcript, transcript_manifest)
        transcript_list = list(res)

    return transcript_list, max_seq_length


def convert_audio(entry):
    raw_path, wav_path = entry[0], entry[1]

    sr = DEFAULT_SAMPLE_RATE

    cmd = "sox -G {} -r {} -b 16 -c 1 {}".format(raw_path, sr, wav_path)
    subprocess.call([cmd], shell=True)

    with sf.SoundFile(wav_path) as f:
        samples = len(f)
        sr = f.samplerate
        duration = len(f) / float(sr)

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
    pass

#######################################################################################################################
