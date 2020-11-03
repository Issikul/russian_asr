import glob
import os
import subprocess
import tarfile
import wget
import json
import librosa
from pathlib import Path


WORK_DIR = Path.cwd().parent


def prepare_an4(data_dir: str):
    # Download the dataset. This will take a few moments...
    print("******")
    if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
        an4_url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz'
        an4_path = wget.download(an4_url, data_dir)
        print(f"Dataset downloaded at: {an4_path}")
    else:
        print("Tarfile already exists.")
        an4_path = data_dir + '/an4_sphere.tar.gz'

    if not os.path.exists(data_dir + '/an4/'):
        # Untar and convert .sph to .wav (using sox)
        tar = tarfile.open(an4_path)
        tar.extractall(path=data_dir)

        print("Converting .sph to .wav...")
        sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
        for sph_path in sph_list:
            wav_path = sph_path[:-4] + '.wav'
            cmd = ["sox", sph_path, wav_path]
            subprocess.run(cmd)
    print("Finished conversion.\n******")


def create_manifest_an4(data_dir: str):
    # --- Building Manifest Files --- #

    # Function to build a manifest
    def build_manifest(transcripts_path, manifest_path, wav_path):
        with open(transcripts_path, 'r') as fin:
            with open(manifest_path, 'w') as fout:
                for line in fin:
                    # Lines look like this:
                    # <s> transcript </s> (fileID)
                    transcript = line[: line.find('(') - 1].lower()
                    transcript = transcript.replace('<s>', '').replace('</s>', '')
                    transcript = transcript.strip()

                    file_id = line[line.find('(') + 1: -2]  # e.g. "cen4-fash-b"
                    audio_path = os.path.join(
                        data_dir, wav_path,
                        file_id[file_id.find('-') + 1: file_id.rfind('-')],
                        file_id + '.wav')

                    duration = librosa.core.get_duration(filename=audio_path)

                    # Write the metadata to the manifest
                    metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": transcript
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')

    # Building Manifests
    print("******")
    train_transcripts = data_dir + '/an4/etc/an4_train.transcription'
    train_manifest = data_dir + '/an4/train_manifest.json'
    if not os.path.isfile(train_manifest):
        build_manifest(train_transcripts, train_manifest, 'an4/wav/an4_clstk')
        print("Training manifest created.")

    test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
    test_manifest = data_dir + '/an4/test_manifest.json'
    if not os.path.isfile(test_manifest):
        build_manifest(test_transcripts, test_manifest, 'an4/wav/an4test_clstk')
        print("Test manifest created.")
    print("***Done***")


if __name__ == '__main__':
    prepare_an4(str(WORK_DIR / 'datasets'))
    create_manifest_an4(str(WORK_DIR / 'datasets'))
