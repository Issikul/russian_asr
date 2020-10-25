import os
import tarfile
import subprocess
import librosa
import json
from pathlib import Path


WORK_DIR = Path.cwd().parent


def mainfest_from_librispeech(data_dir: str, f_type: str):
    # recursively walk through give data dir
    with open("{}/{}".format(data_dir, "librispeech_manifest_" + f_type + ".json"), "w") as manifest:
        for root, _, files in os.walk(data_dir):
            # only search in dirs containing files (not only directories)
            if files:
                for file in files:
                    # find txt files
                    if file[-3:] == "txt":
                        with open("{}/{}".format(root, file), "r") as f:
                            for line in f.readlines():

                                path_to_wav = root + "/" + line.split(" ")[0] + ".wav"

                                duration = librosa.core.get_duration(filename=path_to_wav)

                                transcript = str.lower(line[line.find(' ')+1:])
                                transcript = transcript.replace('\n', '')

                                # Write the metadata to the manifest
                                metadata = {
                                    "audio_filepath": path_to_wav,
                                    "duration": duration,
                                    "text": transcript
                                }
                                json.dump(metadata, manifest)
                                manifest.write('\n')


def flac_to_wav_recursively(data_dir: str):
    # recursively walk through give data dir
    for root, _, files in os.walk(data_dir):
        # only search in dirs containing files (not only directories)
        if files:
            for file in files:
                # find flac files
                if file[-4:] == "flac":
                    # convert
                    cmd = ["ffmpeg", "-i", "{}/{}".format(root, file), "-ar", "16000",
                           "{}/{}".format(root, (file[:-4] + "wav"))]
                    subprocess.run(cmd)


def untar_librispeech(data_dir: str, tar_path: str):
    if not os.path.exists(data_dir + '/librispeech/'):
        # Untar and convert .sph to .wav (using sox)
        tar = tarfile.open(tar_path)
        tar.extractall(path=data_dir + '/librispeech/')


if __name__ == '__main__':
    # untar_librispeech(str(WORK_DIR), str(WORK_DIR / 'train-clean-100.tar.gz'))
    # flac_to_wav_recursively(str(WORK_DIR / 'librispeech' / 'train-clean-100'))
    # flac_to_wav_recursively(str(WORK_DIR / 'librispeech' / 'test-clean'))
    mainfest_from_librispeech(str(WORK_DIR / 'datasets' / 'librispeech' / 'train-clean-100'), 'train')
    mainfest_from_librispeech(str(WORK_DIR / 'datasets' / 'librispeech' / 'test-clean'), 'test')
