import glob
import json
import subprocess
import librosa
from pathlib import Path


# Current working directory
WORK_DIR = Path.cwd().parent


def convert_mp3_to_wav(input_dir: Path, output_dir: Path):
    """
    Convert all .mp3 files from given input dir to
    .wav files in given output directory
    :param input_dir:
    :param output_dir:
    :return:
    """

    # Create new dir for .wav files if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir()

    # Create string for glob and fetch list of .mp3 file paths
    mp3_path_string = str(input_dir / '**' / '*.mp3')
    mp3_list = glob.glob(mp3_path_string, recursive=True)

    # Create new .wav file paths and convert files
    for mp3_path in mp3_list:
        wav_path = output_dir / (str(Path(mp3_path).name)[:-4] + '.wav')

        # Create if file doesn't exist
        if not wav_path.exists():
            cmd = ["ffmpeg", "-i", mp3_path, "-ar", "16000", str(wav_path)]
            subprocess.run(cmd)


def change_tsv_path_name(tsv_file: str):
    """
    Change path name, .mp3 in Mozilla Common Voice .tsv files to .wav
    :param tsv_file:
    :return:
    """

    # Output file path
    output_path = "{}{}{}".format(tsv_file[:-4], '_wav', '.tsv')

    with open(output_path, 'w') as outfile:
        with open(tsv_file, 'r') as infile:
            for line in infile.readlines():
                mp3_start = line.find(".mp3")
                mp3_end = mp3_start + 4

                if not mp3_start == -1:
                    formatted_line = '{}{}{}'.format(line[:mp3_start], '.wav', line[mp3_end:])
                else:
                    formatted_line = line

                outfile.write(formatted_line)


def format_transcript(transcript: str) -> str:
    """
    drop "'" "?" "!" "," "." "-" and capital letters
    :param transcript:
    :return:
    """
    transcript = transcript.replace(".", "")
    transcript = transcript.replace("!", "")
    transcript = transcript.replace("?", "")
    transcript = transcript.replace("'", "")
    transcript = transcript.replace(",", "")
    transcript = transcript.replace("-", "")
    transcript = transcript.replace("\"", "")
    transcript = transcript.replace("\\", "")
    transcript = transcript.replace("/", "")
    transcript = transcript.replace("C", "с")
    transcript = transcript.replace("Firefox", "")
    if "Intel" in transcript:
        return ""

    for num, char in enumerate(transcript):
        if 1040 <= ord(char) <= 1071:
            transcript = "{}{}{}".format(transcript[:num], chr((ord(char) + 32)), transcript[num+1:])
        elif not 1040 <= ord(char) <= 1104 and not char == ' ':
            return ""
    return transcript


def create_json_manifest(data_path: Path, transcripts_path: str, manifest_path: str):
    """
    Create .json manifest for NeMo teaching
    :param data_path:
    :param work_dir:
    :param transcripts_path:
    :param manifest_path:
    :return:
    """
    if Path(manifest_path).exists():
        return

    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in fin.readlines()[1:]:
                line = line.split('\t')

                # get relevant data for .json
                audio_path = str(data_path / "clips_wav" / line[1])

                duration = librosa.core.get_duration(filename=audio_path)

                transcript = line[2]
                transcript = format_transcript(transcript)
                if transcript == "":
                    continue

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')


def create_custom_json_manifest(data_path: Path, exclude_paths: list, transcripts_path: str, manifest_path: str):
    """
    Create .json manifest for NeMo teaching
    :param exclude_paths:
    :param data_path:
    :param transcripts_path:
    :param manifest_path:
    :return:
    """

    exclude_data = []
    for path in exclude_paths:
        with open(path, 'r') as ex:
            for l in ex.readlines()[1:]:
                exclude_data.append(l.split('\t')[2])

    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for n, l in enumerate(fin.readlines()[1:]):
                l = l.split('\t')

                if l[2] in exclude_data:
                    continue

                # get relevant data for .json
                audio_path = str(data_path / "clips_wav" / l[1])

                duration = librosa.core.get_duration(filename=audio_path)

                transcript = l[2]
                transcript = format_transcript(transcript)
                if transcript == "":
                    continue

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')


if __name__ == '__main__':
    data_dir = WORK_DIR / 'mozilla' / 'ru'

    # 1, wav rewrite
    convert_mp3_to_wav(data_dir / 'clips', data_dir / 'clips_wav')

    # 2, rewrite paths for new .wav names
    # path_rewrite = ['dev.tsv', 'invalidated.tsv', 'validated.tsv', 'train.tsv', 'test.tsv', 'reported.tsv']
    # for file_name in path_rewrite:
    #    change_tsv_path_name(str(data_dir / file_name))

    # 3, create .json file(s) for NeMo toolkit
    # wav_data = ['dev_wav.tsv', 'invalidated_wav.tsv', 'validated_wav.tsv',
    #            'train_wav.tsv', 'test_wav.tsv']
    # manifest_data = ['dev.json', 'invalidated.json', 'validated.json',
    #                 'train.json', 'test.json']
    # for i in range(len(wav_data)):
    #    create_json_manifest(data_dir, str(data_dir / wav_data[i]), str(data_dir / manifest_data[i]))

    # 4
    create_custom_json_manifest(data_dir, [str(data_dir / 'dev_wav.tsv'), str(data_dir / 'test_wav.tsv')],
                                str(data_dir / 'validated_wav.tsv'), str(data_dir / 'custom_train.json'))
