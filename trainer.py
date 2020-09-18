# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from ruamel.yaml import YAML
from omegaconf import DictConfig
from pathlib import Path


WORK_DIR = Path.cwd()


def train():
    # Set config and manifest paths
    config_path = str(WORK_DIR / "cv-corpus-5.1-2020-06-22" / "ru" / "configs" / "config.yaml")
    train_manifest = str(WORK_DIR / "cv-corpus-5.1-2020-06-22" / "ru" / "train.json")
    test_manifest = str(WORK_DIR / "cv-corpus-5.1-2020-06-22" / "ru" / "test.json")

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    print(params)

    trainer = pl.Trainer(gpus=1, max_epochs=5)

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # Start training!!!
    trainer.fit(first_asr_model)


if __name__ == '__main__':
    train()
