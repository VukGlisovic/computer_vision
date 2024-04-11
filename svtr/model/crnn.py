import torch
from torch import nn


class CRNN(nn.Module):

    def __init__(self,
                 img_shape=[3, 32, 100],
                 vocab_size=11):
        super().__init__()
        self.img_shape = img_shape
        self.vocab_size = vocab_size

        encoder_in_c = self.img_shape[0]
        encoder_out_c = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_in_c, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),
            nn.Conv2d(512, encoder_out_c, kernel_size=2, padding='same', bias=False),
            nn.ReLU()
        )

        lstm_in_c = 256
        lstm_out_c = 256
        self.time_dense = nn.Linear(encoder_out_c, lstm_in_c)
        self.time_act = nn.ReLU()
        self.lstm1 = nn.LSTM(lstm_in_c, 256, batch_first=True, bidirectional=True)  # the 2 directions will be summed
        self.lstm2 = nn.LSTM(256, lstm_out_c, batch_first=True, bidirectional=True)  # the 2 directions will be concatenated
        self.out_dense = nn.Linear(2*lstm_out_c, vocab_size)

    def forward(self, x):
        x = self.encoder(x)  # out shape: [bs, C, H, W]
        bs = x.shape[0]
        c = x.shape[1]
        x = x.reshape((bs, c, -1))  # out shape: [bs, C, H*W]
        x = x.permute((0, 2, 1))  # out shape: [bs, H*W, C]
        x = self.time_dense(x)
        x = self.time_act(x)
        x, (h_n, c_n) = self.lstm1(x)
        x = x[:, :, :256] + x[:, :, 256:]
        x, (h_n, c_n) = self.lstm2(x)  # out shape: [bs, H*W, C]
        x = self.out_dense(x)  # out shape: [bs, H*W, n_characters]
        out = x.log_softmax(axis=2)
        return out
