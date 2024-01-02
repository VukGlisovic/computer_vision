import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_loss(model, dl):
    """Evaluates the performance of the model on the provided dataloader.

    Args:
        model (Model): pytorch model
        dl (Dataloader): pytorch dataloader

    Returns:
        float
    """
    # set the model to evaluation mode
    model.eval()

    losses = []
    for x, y in dl:
        _, loss = model(x, y)
        losses.append(loss.item())

    return np.mean(losses)


def train(model, optimizer, dl_train, dl_val, n_epochs, scheduler=None):
    """Trains a model on the training dataloader while also evaluating the model
    on the validation dataloader at the end of each epoch.

    Args:
        model (Model): pytorch model
        optimizer (Optimizer): pytorch optimizer
        dl_train (Dataloader): training pytorch dataloader
        dl_val (Dataloader): validation pytorch dataloader
        n_epochs (int):
        scheduler (Scheduler): pytorch scheduler

    Returns:
        pd.DataFrame: contains resulting metrics
    """
    # set model in training mode
    model.train()

    # Placeholder for storing losses
    metrics = {'train': [], 'val': []}
    if scheduler:
        metrics['lr'] = []

    # Iterate through epochs
    for epoch in range(n_epochs):

        train_losses = []
        for x, y in (pbar := tqdm(dl_train)):
            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass through the model to calculate logits and loss
            logits, loss = model(x, targets=y)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            loss_train = loss.item()
            train_losses.append(loss_train)
            pbar.set_description(f"Ep {epoch + 1}/{n_epochs} | Train loss {loss_train:.3f}")

        metrics['train'].append(np.mean(train_losses))

        # adjust the learning rate if there is a lr scheduler
        if scheduler:
            metrics['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()

        # evaluate loss on the validation set
        loss_val = evaluate_loss(model, dl_val)
        metrics['val'].append(loss_val)
        print(f"Ep {epoch + 1}/{n_epochs} | Train loss {metrics['train'][-1]:.3f} | Val loss {loss_val:.3f}")

    return pd.DataFrame(metrics)
