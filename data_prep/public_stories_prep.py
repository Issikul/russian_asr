import os
import subprocess
import json
import librosa
from typing import List
from pathlib import Path


WORK_DIR = Path.cwd().parent


def mainfest_from_public_stories(data_dir: str, f_type: str):
    # recursively walk through give data dir
    with open("{}/{}".format(data_dir, f_type + ".json"), "w") as manifest:
        for root, _, files in os.walk(data_dir):
            # only search in dirs containing files (not only directories)
            if files:
                for file in files:
                    # find txt files
                    if file[-3:] == "txt":
                        path_to_wav = root + '/' + file[:-3] + 'wav'
                        with open("{}/{}".format(root, file), "r") as f:
                            for line in f.readlines():

                                duration = librosa.core.get_duration(filename=path_to_wav)

                                transcript = line
                                transcript = transcript.replace('\n', '')

                                # Write the metadata to the manifest
                                metadata = {
                                    "audio_filepath": path_to_wav,
                                    "duration": duration,
                                    "text": transcript
                                }
                                json.dump(metadata, manifest, ensure_ascii=False)
                                manifest.write('\n')


def opus_to_wav_recursively(data_dir: str):
    # recursively walk through give data dir
    for root, _, files in os.walk(data_dir):
        # only search in dirs containing files (not only directories)
        if files:
            for file in files:
                # find flac files
                if file[-4:] == "opus":
                    # convert
                    cmd = ["ffmpeg", "-i", "{}/{}".format(root, file), "-ar", "16000",
                           "{}/{}".format(root, (file[:-4] + "wav"))]
                    subprocess.run(cmd)


def fuse_manifests(manifest_paths: List[str], combine_path: str):
    with open(combine_path, 'w') as new_manifest:
        for manifest_path in manifest_paths:
            with open(manifest_path, 'r') as manifest:
                for line in manifest.readlines():
                    line = json.loads(line)

                    duration = line['duration']
                    transcript = line['text']
                    path_to_wav = line['audio_filepath']

                    # Write the metadata to the manifest
                    metadata = {
                        "audio_filepath": path_to_wav,
                        "duration": duration,
                        "text": transcript
                    }
                    json.dump(metadata, new_manifest, ensure_ascii=False)
                    new_manifest.write('\n')


if __name__ == '__main__':
    # convert opus to wav
    # opus_to_wav_recursively(str(WORK_DIR / 'asr_public_stories_ru'))

    # opus_to_wav_recursively(str(WORK_DIR / 'datasets' / 'radio_2'))

    # create manifest
    mainfest_from_public_stories(str(WORK_DIR / 'datasets' / 'radio_2'), 'train_manifest')

    # fuse given manifests
    # manifests_to_fuse = [str(WORK_DIR / 'datasets' / 'public_youtube700_val' / 'dev_manifest.json'),
    #                     str(WORK_DIR / 'datasets' / 'buriy_audiobooks_2_val' / 'dev_manifest.json')]
    # fuse_manifests(manifest_paths=manifests_to_fuse, combine_path=str(WORK_DIR / 'ru_buriy_yt.json'))
