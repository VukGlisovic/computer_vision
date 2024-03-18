import os
import argparse

import torch
from torch.utils.data import DataLoader

from svtr.data_pipeline.mnist import ConcatenatedMNISTDataset
from svtr.model.model import SVTR, print_model_parameters
from svtr.model.ctc_decoder import CTCDecoder
from svtr.model.training import train
from svtr.constants import EXPERIMENTS_DIR


def main(architecture='tiny'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    # create train and test dataloaders
    dataset_train = ConcatenatedMNISTDataset(num_digits=5, train=True, device=device)
    dataset_test = ConcatenatedMNISTDataset(num_digits=5, train=False, device=device)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)
    # create model and corresponding decoder
    model_svtr = SVTR(architecture=architecture, img_shape=[1, 32, 160], vocab_size=dataset_train.vocab_size)
    model_svtr = model_svtr.to(device)
    decoder = CTCDecoder(dataset_train.vocab)
    print_model_parameters(model_svtr)
    # create optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model_svtr.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # create checkpoint directory
    output_dir = os.path.join(EXPERIMENTS_DIR, f'svtr_{architecture}')
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoints_dir, 'svtr_ep{epoch:02d}.pth')
    os.makedirs(checkpoints_dir, exist_ok=True)
    # execute main training function
    train(
        model_svtr,
        decoder,
        optimizer,
        dataloader_train,
        dataloader_test,
        n_epochs=5,
        scheduler=lr_scheduler,
        ckpt_path=checkpoint_path,
        output_dir=output_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--architecture', type=str, default='tiny',
                        help="Choose the SVTR architecture size. Options are: tiny, small, base and large.")
    known_args, _ = parser.parse_known_args()
    main(known_args.architecture)
