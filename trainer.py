import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER
import pytorch_lightning as pl
from nemo.core import ModelPT
from ruamel.yaml import YAML
from omegaconf import DictConfig
from pathlib import Path

WORK_DIR = Path.cwd()


def train():
    # Set config and manifest paths
    config_path = str(WORK_DIR / "configs" / "config_russian.yaml")
    train_manifest = str(WORK_DIR / "mozilla" / "ru" / "custom_train.json")
    test_manifest = str(WORK_DIR / "mozilla" / "ru" / "test.json")

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    trainer = pl.Trainer(gpus=1, max_epochs=200)

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest

    first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # Start training!!!
    trainer.fit(first_asr_model)


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
    asr_model = asr_model.load_from_checkpoint(str(WORK_DIR / "lightning_logs" / "version_31" /
                                                   "checkpoints" / "epoch=81.ckpt"))
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


def train_an4(work_dir: Path):
    # --- Config Information ---#
    config_path = str(work_dir / 'configs' / 'config_an4.yaml')
    train_manifest = str(work_dir / 'an4' / 'train_manifest.json')
    test_manifest = str(work_dir / 'an4' / 'test_manifest.json')

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    trainer = pl.Trainer(gpus=1, max_epochs=200)

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    an4_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # Start training!!!
    trainer.fit(an4_asr_model)


def train_librispeech(work_dir: Path, ckpt=None):
    # --- Config Information ---#
    config_path = str(work_dir / 'configs' / 'config_librispeech_2.yaml')
    train_manifest = str(work_dir / 'librispeech' / 'librispeech_manifest_train.json')
    test_manifest = str(work_dir / 'librispeech' / 'librispeech_manifest_test.json')

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    # ADD RESUME FROM CHECKPOINT, NOT ONLY LOAD FROM CHECKPOINT
    trainer = pl.Trainer(gpus=1, max_epochs=200, resume_from_checkpoint=ckpt)

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest

    ls_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # If loading from checkpoint
    if ckpt:
        ls_asr_model = ls_asr_model.load_from_checkpoint(checkpoint_path=ckpt, hparams_file=
                                                         str(WORK_DIR / "lightning_logs" / "version_34" /
                                                             "hparams.yaml"))

        ls_asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
        ls_asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])


    # Start training!!!
    trainer.fit(ls_asr_model)


if __name__ == '__main__':
    # train librispeech
    train_librispeech(WORK_DIR, str(WORK_DIR / "lightning_logs" / "version_44" / "checkpoints" / "epoch=97.ckpt"))
    # train_librispeech(WORK_DIR)

    # train an4
    # train_an4(WORK_DIR)

    # train()
    # inference()
    # check_duration(str(WORK_DIR / "mozilla" / "ru" / "test.json"))
