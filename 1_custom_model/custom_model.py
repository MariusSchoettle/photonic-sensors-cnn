import torch
from torch import nn


class CustomConvModel(nn.Module):
    """
    Convolutional neural network for training on TTI images. Takes in number of hidden_units per layer and the
    input shape of each sample to calculate the length of the flattened layer in the end.
    """

    def __init__(self, hidden_units, input_shape, num_blocks=4, dropout_prob=0.5):
        super().__init__()

        conv_blocks = []
        in_channels = input_shape[0]
        for _ in range(num_blocks):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, hidden_units, kernel_size=3, padding=1, stride=1
                    ),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        hidden_units, hidden_units, kernel_size=3, padding=1, stride=1
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2),
                )
            )
            in_channels = hidden_units

        self.conv = nn.Sequential(*conv_blocks, nn.AdaptiveAvgPool2d((1, 1)))

        # Determine length of flattened output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.conv(dummy_input)
            flattened_dim = out.view(1, -1).shape[1]

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_dim, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, x):
        return self.regression(self.conv(x))
