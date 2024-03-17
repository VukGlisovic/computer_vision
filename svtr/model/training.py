import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from svtr.model.model import save_model
from svtr.model.ctc_loss import CTCLoss


@torch.no_grad()
def evaluate_loss(model, loss_fnc, dl):
    """Evaluates the performance of the model on the provided dataloader.

    Args:
        model (Model): pytorch model
        loss_fnc (Callable):
        dl (Dataloader): pytorch dataloader

    Returns:
        float
    """
    # set the model to evaluation mode
    model.eval()

    losses = []
    for x, y in tqdm(dl):
        pred = model(x)
        loss = loss_fnc(pred, y)
        losses.append(loss.item())

    return np.mean(losses)


def train(model, optimizer, dl_train, dl_val, n_epochs, scheduler=None, ckpt_path=None):
    """Trains a model on the training dataloader while also evaluating the model
    on the validation dataloader at the end of each epoch.

    Args:
        model (Model): pytorch model
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
    # Placeholder for storing losses
    metrics = {'train_loss': [], 'val_loss': [], 'train_ned': [], 'val_ned': []}
    if scheduler:
        metrics['lr'] = []

    # Iterate through epochs
    for epoch in range(n_epochs):
        # set model in training mode
        model.train()

        train_losses = []
        for x, y in (pbar := tqdm(dl_train)):
            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass through the model to calculate logits and loss
            logits = model(x)
            loss = ctc_loss(logits, y)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            loss_train = loss.item()
            train_losses.append(loss_train)
            pbar.set_description(f"Ep {epoch + 1}/{n_epochs} | Train loss {np.mean(train_losses):.4f}")

        metrics['train_loss'].append(np.mean(train_losses))

        # adjust the learning rate if there is a lr scheduler
        if scheduler:
            metrics['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()

        # evaluate loss on the validation set
        val_loss = evaluate_loss(model, ctc_loss, dl_val)
        metrics['val_loss'].append(val_loss)
        print(f"Ep {epoch + 1}/{n_epochs} | Train loss {metrics['train_loss'][-1]:.3f} | Val loss {val_loss:.3f}")

        # save model checkpoint if checkpoint path configured
        if ckpt_path:
            save_model(model, ckpt_path.format(epoch=epoch))

    return pd.DataFrame(metrics)
