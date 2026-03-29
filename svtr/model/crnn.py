from typing import Tuple

from torch import nn


class CRNN(nn.Module):

    def __init__(self,
                 image_shape: Tuple[int, int, int] = None,
                 vocab_size: int = 11):
        super().__init__()
        self.image_shape = image_shape
        self.vocab_size = vocab_size

        encoder_in_c = self.image_shape[0]
        encoder_out_c = 512
        self.first_conv_nr_filters = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_in_c, self.first_conv_nr_filters, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(self.first_conv_nr_filters, 2 * self.first_conv_nr_filters, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(2 * self.first_conv_nr_filters, 4 * self.first_conv_nr_filters, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(4 * self.first_conv_nr_filters),
            nn.ReLU(),
            nn.Conv2d(4 * self.first_conv_nr_filters, 4 * self.first_conv_nr_filters, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),

            nn.Conv2d(4 * self.first_conv_nr_filters, 8 * self.first_conv_nr_filters, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(8 * self.first_conv_nr_filters),
            nn.ReLU(),
            nn.Conv2d(8 * self.first_conv_nr_filters, 8 * self.first_conv_nr_filters, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),

            nn.Conv2d(8 * self.first_conv_nr_filters, encoder_out_c, kernel_size=2, padding='same', bias=False),
            nn.BatchNorm2d(8 * self.first_conv_nr_filters),
            nn.ReLU()
        )

        self.lstm_in_c = 128
        self.lstm_out_c = 256

        self.time_dense = nn.Linear(encoder_out_c, self.lstm_in_c)
        self.time_act = nn.ReLU()
        self.lstm1 = nn.LSTM(self.lstm_in_c, self.lstm_out_c, batch_first=True, bidirectional=True)  # the 2 directions will be summed
        self.lstm2 = nn.LSTM(self.lstm_out_c, self.lstm_out_c, batch_first=True, bidirectional=True)  # the 2 directions will be concatenated
        self.out_dense = nn.Linear(2*self.lstm_out_c, self.vocab_size)

    def forward(self, x):
        x = self.encoder(x)  # out shape: [bs, C, H, W]
        bs = x.shape[0]
        c = x.shape[1]
        x = x.reshape((bs, c, -1))  # out shape: [bs, C, H*W]
        x = x.permute((0, 2, 1))  # out shape: [bs, H*W, C]
        x = self.time_dense(x)
        x = self.time_act(x)
        x, (h_n, c_n) = self.lstm1(x)
        x = x[:, :, :self.lstm_out_c] + x[:, :, self.lstm_out_c:]
        x, (h_n, c_n) = self.lstm2(x)  # out shape: [bs, H*W, C]
        x = self.out_dense(x)  # out shape: [bs, H*W, n_characters]
        out = x.log_softmax(axis=2)
        return out
