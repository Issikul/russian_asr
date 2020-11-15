import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER, word_error_rate
import pytorch_lightning as pl
import torch
from ruamel.yaml import YAML
from omegaconf import DictConfig
from pathlib import Path

WORK_DIR = Path.cwd()

# russian train settings
config_path_ru = str(WORK_DIR / "configs" / "config_russian_12x1_lr_001_short.yaml")
train_manifest_ru = str(WORK_DIR / "datasets" / "mozilla" / "ru" / "custom_train.json")
test_manifest_ru = str(WORK_DIR / "datasets" / "mozilla" / "ru" / "test.json")
dev_manifest_ru = str(WORK_DIR / "datasets" / "mozilla" / "ru" / "test.json")

# librispeech clean 100 settings
config_path_ls = str(WORK_DIR / 'configs' / 'quartznet12x1.yaml')
train_manifest_ls = str(WORK_DIR / "datasets" / 'librispeech' / 'librispeech_manifest_train.json')
test_manifest_ls = str(WORK_DIR / "datasets" / 'librispeech' / 'librispeech_manifest_test.json')

# an4 settings
config_path_an4 = str(WORK_DIR / 'configs' / 'quartznet12x1.yaml')
train_manifest_an4 = str(WORK_DIR / "datasets" / 'an4' / 'train_manifest.json')
test_manifest_an4 = str(WORK_DIR / "datasets" / 'an4' / 'test_manifest.json')


def train_model(config_path: str, train_manifest: str, test_manifest: str, dev_manifest: str, checkpoint: str = None) \
        -> nemo_asr.models.EncDecCTCModel:
    # Set config
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    # Add train and test paths to config
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    params['model']['test_ds']['manifest_filepath'] = dev_manifest

    # Setup trainer
    trainer = pl.Trainer(gpus=1, max_epochs=100, resume_from_checkpoint=checkpoint)

    if checkpoint == "pretrained":
        # Load pretrained encoder 15x5
        # asr_model_en = nemo_asr.models.EncDecCTCModel.from_pretrained("QuartzNet15x5Base-En")

        # Create uninitialized model
        asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

        # Set encoder weights to pretrained
        asr_model.encoder.load_state_dict(torch.load(
            '/home/geripc/gitrepos/russian_asr/pretrained/JasperEncoder-STEP-174000.pt'))

        # Same for 15x5
        # asr_model.encoder = asr_model_en.encoder

        # Guess its needed...
        asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
        asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])
        asr_model.setup_test_data(test_data_config=params['model']['test_ds'])
    elif checkpoint:
        # Load checkpoint if specified
        asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint)

        # Set low LR, but higher than normal
        # asr_model.cfg['optim']['lr'] = 0.001

        # Guess its needed...
        asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
        asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])
        asr_model.setup_test_data(test_data_config=params['model']['test_ds'])

        # Important?
        # asr_model.cfg = DictConfig(params['model'])

        # Important!
        asr_model.set_trainer(trainer)

        for elem in asr_model.cfg:
            print(elem)
    else:
        # Setup model
        asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    asr_model.summarize()

    # Check whether the model can safely be tested
    print('Safe for testing: {}.'.format(asr_model.prepare_test(trainer)))

    # Start training!!!
    trainer.fit(asr_model)

    return asr_model


def test_model(config_path: str, train_manifest: str, test_manifest: str, checkpoint: str = None):
    """
    Calculates exact WER for checkpoint as seen in its corresponding tensorboard log
    :param config_path:
    :param train_manifest:
    :param test_manifest:
    :param checkpoint:
    :return:
    """
    # Set config
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    # Add train and test paths to config
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    params['model']['test_ds']['manifest_filepath'] = test_manifest

    # Setup model
    # Load checkpoint if specified
    if checkpoint:
        asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint)
        asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
        asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])
        asr_model.setup_test_data(test_data_config=params['model']['test_ds'])
    else:
        return

    # Start testing!
    trainer = pl.Trainer(gpus=1)
    trainer.test(asr_model, test_dataloaders=asr_model._test_dl)


def inference(config_path: str, test_manifest: str, checkpoint: str = None):
    """
    Inference for formated data
    :param config_path:
    :param test_manifest:
    :param checkpoint:
    :return:
    """
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    # Bigger batch-size = bigger throughput
    params['model']['validation_ds']['batch_size'] = 1

    # Setup the test data loader and make sure the model is on GPU
    # asr_model.restore_from(restore_path=str(WORK_DIR / 'checkpoint.nemo'))
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint)
    asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])

    asr_model.cuda()

    wer = WER(vocabulary=asr_model.decoder.vocabulary)

    for test_batch in asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        log_probs, encoded_len, greedy_predictions = asr_model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )

        pred = wer.ctc_decoder_predictions_tensor(greedy_predictions)
        trans = wer.ctc_decoder_predictions_tensor(test_batch[2])
        print("original: {}, prediction: {}".format(trans, pred))


def calc_wer(config_path: str, test_manifest: str, checkpoint: str):
    # Get config
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    params['model']['test_ds']['manifest_filepath'] = test_manifest

    # Setup model
    if checkpoint[-4:] == 'nemo':
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=checkpoint)
    else:
        asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint)

    asr_model.setup_test_data(test_data_config=params['model']['test_ds'])

    # Bigger batch-size = bigger throughput
    params['model']['validation_ds']['batch_size'] = 16

    # Make sure the model is on GPU
    asr_model.cuda()

    # We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
    # WER is computed as numerator/denominator.
    # We'll gather all the test batches' numerators and denominators.
    wer_nums = []
    wer_denoms = []

    # Loop over all test batches.
    # Iterating over the model's `test_dataloader` will give us:
    # (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
    # See the AudioToCharDataset for more details.
    for test_batch in asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]
        log_probs, encoded_len, greedy_predictions = asr_model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )
        # Notice the model has a helper object to compute WER
        wer_num, wer_denom = asr_model._wer(greedy_predictions, targets, targets_lengths)
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

    # We need to sum all numerators and denominators first. Then divide.
    print(f"WER = {sum(wer_nums) / sum(wer_denoms)}")


def run_transcription(checkpoint: str = None, json_path: str = None):
    """
    Inference for sound files, also calculates WER of all .wav file paths in given .json file
    :param checkpoint:
    :param json_path:
    :return:
    """
    import json

    # Load ASR model from checkpoint
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint)

    # Load file paths and original transcriptions
    audio_files = []
    transcriptions = []
    with open(json_path, 'r') as json_file:
        for line in json_file.readlines():
            line = json.loads(line)
            audio_files.append(line['audio_filepath'])
            transcriptions.append(line['text'])

    # Calculate predictions
    hypothesis = asr_model.transcribe(paths2audio_files=audio_files, batch_size=16)

    # Print predictions VS original transcriptions
    for num, hypo in enumerate(hypothesis):
        print('{} VS {}'.format(hypo, transcriptions[num]))

    # Calculate WER
    print(f"WER = {word_error_rate(hypothesis, transcriptions)}")


if __name__ == '__main__':
    # train, choose settings from predifined ones, optionally add checkpoint path
    model = train_model(config_path_ru, train_manifest_ru, test_manifest_ru, dev_manifest_ru, str(WORK_DIR / 'lightning_logs'
                                                                        / 'version_98_r2_large_12x1_pretrained2' /
                                                                        'checkpoints' / 'epoch=62.ckpt'))

    model.save_to('latest_model_ru_med_15x5.nemo')

    # inference model , str(WORK_DIR / 'lightning_logs' / 'version_55' /
    #                                                                          'checkpoints' / 'epoch=16.ckpt')
    # inference(config_path_ru, train_manifest_ru, str(WORK_DIR / 'lightning_logs' / 'version_57' /
    #                                                                         'checkpoints' / 'epoch=198.ckpt'))

    # calc WER
    # calc_wer(config_path_ru, test_manifest_ru, str(WORK_DIR / 'lightning_logs' / 'version_69_ru_large_164' /
    #                                        'checkpoints' / 'epoch=97.ckpt'))

    # test
    # test_model(config_path_ru, train_manifest_ru, test_manifest_ru, str(WORK_DIR / 'lightning_logs' / 'version_57' /
    #                                                                         'checkpoints' / 'epoch=198.ckpt'))

    # transcribe
    # run_transcription(checkpoint=str(WORK_DIR / 'lightning_logs' / 'version_81' /
    #                                        'checkpoints' / 'epoch=8.ckpt'), json_path=dev_manifest_ru)
