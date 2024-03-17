import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from svtr.model.model import save_model
from svtr.model.ctc_loss import CTCLoss
from svtr.model.metrics import NormalizedEditDistance
from svtr.model.ctc_decoder import CTCDecoder


@torch.no_grad()
def evaluate_loss(model, dl, loss_fnc, normalized_edit_distance):
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

    losses = []
    ned = []
    for x, y in tqdm(dl):
        pred = model(x)
        loss = loss_fnc(pred, y)
        losses.append(loss.item())
        ned.append(normalized_edit_distance(pred, y))

    return np.mean(losses), np.mean(ned)


def train(model, ctc_decoder, optimizer, dl_train, dl_val, n_epochs, scheduler=None, ckpt_path=None):
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

    Returns:
        pd.DataFrame: contains resulting metrics
    """
    # create loss function
    ctc_loss = CTCLoss(blank=0)
    normalized_edit_distance = NormalizedEditDistance(ctc_decoder)
    # Placeholder for storing losses
    metrics = {'train_loss': [], 'val_loss': [], 'train_ned': [], 'val_ned': []}
    if scheduler:
        metrics['lr'] = []

    # Iterate through epochs
    for epoch in range(n_epochs):
        # set model in training mode
        model.train()

        train_losses = []
        train_ned = []
        for x, y in (pbar := tqdm(dl_train)):
            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass through the model to calculate logits and loss
            logits = model(x)
            loss = ctc_loss(logits, y)
            ned = normalized_edit_distance(logits, y)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            loss_train = loss.item()
            train_losses.append(loss_train)
            train_ned.append(ned)
            pbar.set_description(f"Ep {epoch + 1}/{n_epochs} | Train loss {np.mean(train_losses):.4f} | Train ned {np.mean(train_ned):.4f}")

        metrics['train_loss'].append(np.mean(train_losses))
        metrics['train_ned'].append(np.mean(train_ned))

        # adjust the learning rate if there is a lr scheduler
        if scheduler:
            metrics['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()

        # evaluate loss on the validation set
        val_loss, val_ned = evaluate_loss(model, dl_val, ctc_loss, normalized_edit_distance)
        metrics['val_loss'].append(val_loss)
        metrics['val_ned'].append(val_ned)
        print(f"Ep {epoch + 1}/{n_epochs} "
              f"| Train loss {metrics['train_loss'][-1]:.4f} | Train ned {metrics['train_ned'][-1]:.4f}"
              f"| Val loss {val_loss:.4f} | Val ned {val_ned:.4f}")

        # save model checkpoint if checkpoint path configured
        if ckpt_path:
            save_model(model, ckpt_path.format(epoch=epoch))

    return pd.DataFrame(metrics)
