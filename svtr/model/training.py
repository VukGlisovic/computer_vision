import os.path

import pandas as pd
import torch
from tqdm import tqdm

from svtr.model.model import save_model
from svtr.model.ctc_loss import CTCLoss
from svtr.model.metrics import NormalizedEditDistance
from svtr.model.ctc_decoder import CTCDecoder


@torch.no_grad()
def evaluate_metrics(model, dl, loss_fnc, normalized_edit_distance):
    """Evaluates the performance of the model on the provided dataloader.

    Args:
        model (Model): pytorch model
        dl (Dataloader): pytorch dataloader
        loss_fnc (Callable):
        normalized_edit_distance (Callable):

    Returns:
        float
    """
    # set the model to evaluation mode
    model.eval()

    for x, y in tqdm(dl):
        pred = model(x)
        # update state in loss/metric objects
        loss_fnc(pred, y)
        normalized_edit_distance(pred, y)


def train(model, ctc_decoder, optimizer, dl_train, dl_val, n_epochs, scheduler=None, ckpt_path=None, output_dir=None):
    """Trains a model on the training dataloader while also evaluating the model
    on the validation dataloader at the end of each epoch.

    Args:
        model (Model): pytorch model
        ctc_decoder (CTCDecoder):
        optimizer (Optimizer): pytorch optimizer
        dl_train (Dataloader): training pytorch dataloader
        dl_val (Dataloader): validation pytorch dataloader
        n_epochs (int):
        scheduler (Scheduler): pytorch scheduler
        ckpt_path (str): template where to save the model. E.g. model-ep{epoch:02d}.pth
        output_dir (str):

    Returns:
        pd.DataFrame: contains resulting metrics
    """
    # create loss function
    ctc_loss = CTCLoss(blank=0)
    # create metrics
    normalized_edit_distance = NormalizedEditDistance(ctc_decoder)
    # placeholder for storing losses and metrics
    metrics = {'train_loss': [], 'val_loss': [], 'train_ned': [], 'val_ned': [], 'train_acc': [], 'val_acc': []}
    if scheduler:
        metrics['lr'] = []

    # start training
    for epoch in range(n_epochs):
        # set model in training mode
        model.train()

        for x, y in (pbar := tqdm(dl_train)):
            # zero out gradients
            optimizer.zero_grad()

            # forward pass through the model to calculate log_softmax output and loss
            logits = model(x)
            loss = ctc_loss(logits, y)
            normalized_edit_distance(logits, y)

            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            # update progress bar
            pbar.set_description(f"Ep {epoch + 1}/{n_epochs} "
                                 f"| Train loss {ctc_loss.compute():.4f} "
                                 f"| Train ned/acc {normalized_edit_distance.ned_result():.4f}/{normalized_edit_distance.acc_result()*100:.2f}")

        metrics['train_loss'].append(ctc_loss.compute())
        metrics['train_ned'].append(normalized_edit_distance.ned_result())
        metrics['train_acc'].append(normalized_edit_distance.acc_result())
        ctc_loss.reset()
        normalized_edit_distance.reset()

        # adjust the learning rate if there is a lr scheduler
        if scheduler:
            metrics['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()

        # evaluate loss on the validation set
        evaluate_metrics(model, dl_val, ctc_loss, normalized_edit_distance)
        metrics['val_loss'].append(ctc_loss.compute())
        metrics['val_ned'].append(normalized_edit_distance.ned_result())
        metrics['val_acc'].append(normalized_edit_distance.acc_result())
        ctc_loss.reset()
        normalized_edit_distance.reset()
        print(f"Ep {epoch + 1}/{n_epochs} "
              f"| Train loss {metrics['train_loss'][-1]:.4f} | Train ned/acc {metrics['train_ned'][-1]:.4f}/{metrics['train_acc'][-1]*100:.2f}"
              f"| Val loss {metrics['val_loss'][-1]:.4f} | Val ned/acc {metrics['val_ned'][-1]:.4f}/{metrics['val_acc'][-1]*100:.2f}")

        # save model checkpoint if checkpoint path configured
        if ckpt_path:
            save_model(model, ckpt_path.format(epoch=epoch))

    df_metrics = pd.DataFrame(metrics)
    if output_dir:
        df_metrics.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    return df_metrics
