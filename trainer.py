import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER
import pytorch_lightning as pl
from ruamel.yaml import YAML
from omegaconf import DictConfig
from pathlib import Path


WORK_DIR = Path.cwd()


# russian train settings
config_path_ru = str(WORK_DIR / "configs" / "config_russian_15x5.yaml")
train_manifest_ru = str(WORK_DIR / "mozilla" / "ru" / "train.json")
test_manifest_ru = str(WORK_DIR / "mozilla" / "ru" / "test.json")


# librispeech clean 100 settings
config_path_ls = str(WORK_DIR / 'configs' / 'quartznet12x1.yaml')
train_manifest_ls = str(WORK_DIR / 'librispeech' / 'librispeech_manifest_train.json')
test_manifest_ls = str(WORK_DIR / 'librispeech' / 'librispeech_manifest_test.json')


# an4 settings
config_path_an4 = str(WORK_DIR / 'configs' / 'quartznet12x1.yaml')
train_manifest_an4 = str(WORK_DIR / 'an4' / 'train_manifest.json')
test_manifest_an4 = str(WORK_DIR / 'an4' / 'test_manifest.json')


def train_model(config_path: str, train_manifest: str, test_manifest: str, checkpoint: str = None):
    # Set config
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    # Add train and test paths to config
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest

    # Setup trainer
    trainer = pl.Trainer(gpus=1, max_epochs=200, resume_from_checkpoint=checkpoint)

    # Setup model
    asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # Load checkpoint if specified
    if checkpoint:
        # ADD if NEEDED: hparams_file=str(WORK_DIR / "lightning_logs" / "version_34" / "hparams.yaml")
        asr_model = asr_model.load_from_checkpoint(checkpoint_path=checkpoint)

        asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
        asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])

    # Start training!!!
    trainer.fit(asr_model)


def inference():
    # Set config and manifest paths
    config_path = str(WORK_DIR / "configs" / "config_russian.yaml")
    train_manifest = str(WORK_DIR / "mozilla" / "ru" / "train.json")
    test_manifest = str(WORK_DIR / "mozilla" / "ru" / "test.json")

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    # Bigger batch-size = bigger throughput
    params['model']['validation_ds']['batch_size'] = 1
    trainer = pl.Trainer(gpus=1, max_epochs=100)

    asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # Setup the test data loader and make sure the model is on GPU
    # asr_model.restore_from(restore_path=str(WORK_DIR / 'checkpoint.nemo'))
    asr_model = asr_model.load_from_checkpoint(str(WORK_DIR / "lightning_logs" / "version_51" /
                                                   "checkpoints" / "epoch=194.ckpt"))
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


def check_duration(file_path):
    import json
    overall = 0
    with open(file_path, encoding='utf-8') as json_file:
        data = json.load(json_file)
        for d in data['duration']:
            overall += float(d)

    print(overall)


if __name__ == '__main__':
    # train, choose settings from predifined ones, optionally add checkpoint path
    train_model(config_path_an4, train_manifest_an4, test_manifest_an4)

    # inference model
    # inference()

    # other
    # check_duration(str(WORK_DIR / "mozilla" / "ru" / "test.json"))
