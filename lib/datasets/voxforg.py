import os
import tarfile

import wget

DEFAULT_SAMPLE_RATE = 16000


#######################################################################################################################

def create_manifest(root_path='./data/VOXFORG', manifest_file='manifest.json',
                    urls="https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz", verbose=0):
    vprint = print if verbose > 0 else lambda *a, **k: None

    vprint('Creating VOXFORG dataset in root dir: ', root_path, end='')
    extract_path = os.path.join(root_path, 'extract')
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    vprint(' ...done.')

    urls = urls if isinstance(urls, list) else [urls]

    for url in urls:
        vprint('Tar file download:', url, end='')
        tar_path = os.path.join(extract_path, url.split("/")[-1])
        download_dataset(url, tar_path, extract_path)
        vprint(' ... done.')

        vprint('Tar file xtraction:', tar_path, extract_path, end='')
        extract_tarfile(tar_path, extract_path)
        print(' ...done.')


def download_dataset(url, tar_path, extract_path):
    if not os.path.exists(tar_path):
        wget.download(url, out=extract_path)


def extract_tarfile(tar_path, extract_path):
    if not os.path.exists(tar_path[:-7]):
        tar = tarfile.open(tar_path)
        tar.extractall(extract_path)
        tar.close()

#######################################################################################################################
