import torch


class CTCLoss:

    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        """https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

        Args:
            blank (int):
            reduction (str):
            zero_infinity (bool):
        """
        self.ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.avg_loss = -1
        self.n = 0

    def __call__(self, y_pred, y_true, *args, **kwargs):
        """
        Args:
            y_pred (tensor):
            y_true (tensor):

        Returns:
            tensor: a scalar representing the loss value.
        """
        # y_pred shape: [bs, nr_patches, vocab_size] ([bs, time_steps, nr_classes])
        bs, n_timesteps, n_classes = y_pred.shape
        _, target_length = y_true.shape
        y_pred = y_pred.permute((1, 0, 2))  # CTC loss requires first dim to be timesteps dim
        input_lengths = torch.full(size=(bs,), fill_value=n_timesteps, dtype=torch.long)
        target_lengths = torch.full(size=(bs,), fill_value=target_length, dtype=torch.long)
        loss = self.ctc_loss(y_pred, y_true, input_lengths, target_lengths)
        self.update(loss.item(), bs)
        return loss

    def update(self, loss, bs):
        """Incrementally update the average loss.
        """
        new_n = self.n + bs
        self.avg_loss = self.avg_loss * (self.n / new_n) + loss * (bs / new_n)
        self.n += bs

    def compute(self):
        return self.avg_loss

    def reset(self):
        self.avg_loss = -1
        self.n = 0
