import argparse

import lightning as L
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from srec.data_pipeline.cifar10 import SrecCifar10
from srec.model.lightning_module import SrecLightningModule


def build_dataloaders(data_config):
    dataset_train = SrecCifar10(
        root=data_config['root'],
        train=True,
        horizontal_flip=data_config['horizontal_flip'],
    )
    dataset_val = SrecCifar10(
        root=data_config['root'], 
        train=False,
        horizontal_flip=False,
    )

    num_workers = data_config['num_workers']
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return dataloader_train, dataloader_val


def main(config):
    L.seed_everything(42)

    dataloader_train, dataloader_val = build_dataloaders(config['data'])
    model = SrecLightningModule(**config['model'], **config['optimizer'])

    train_config = config['train']
    trainer = L.Trainer(
        max_epochs=train_config['n_epochs'],
        gradient_clip_val=train_config['gradient_clip_val'],
        default_root_dir=train_config['output_dir'],
        callbacks=[
            ModelCheckpoint(monitor='val/bpsp', mode='min', save_top_k=3, filename='ckpt_ep{epoch:02d}'),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default='config.yaml', help='Path to yaml file.')
    known_args, _ = parser.parse_known_args()
    with open(known_args.config_path, 'r') as f:
        _config = yaml.safe_load(f)
    main(_config)
