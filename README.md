# russian_asr

This is a university project for automatic speech recognition in russian, using the Nvidia NeMo toolkit.

Dataset(s) used for training:
- Mozzila Common Voice, Russian Language (https://drive.google.com/drive/folders/1HJbaEVoK_i__8vqOVwcYyYIiWEfwQUsD?usp=sharing)


Best WER achieved on AN4 dataset using base config modified by:
- lr: 0.02
- weight decay: 0.005
- epoch: 200
- WER: 0.1255 = 12.55 %
