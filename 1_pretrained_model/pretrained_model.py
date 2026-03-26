import torch
from torch import nn
from torchvision import models


class CustomConvModel(nn.Module):
    """
    Pretrained convolutional neural network that is fine-tuned by training on TTI images. Takes in number of
    hidden_units per layer and the input shape of each sample to calculate the length of the flattened layer in the end.
    """

    def __init__(self,
                 input_shape,
                 pretrained_model="efficientnet_b0",
                 pretrained_weights="EfficientNetB0_Weights",
                 freeze_weights=True,
                 dropout_prob=0.5):
        super().__init__()

        # Load the chosen backbone
        backbone_fn = getattr(models, pretrained_model)
        backbone = backbone_fn(weights=f"{pretrained_weights}.DEFAULT")

        self.pool = nn.AdaptiveAvgPool2d(1)

        if hasattr(backbone, "features"):  # e.g. EfficientNet, MobileNet
            self.feature_extractor = backbone.features
        elif hasattr(backbone, "layer4"):  # e.g. ResNet
            self.feature_extractor = nn.Sequential(list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported network architecture: {pretrained_model}.")

        if freeze_weights:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Determine length of flattened output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.pool(self.feature_extractor(dummy_input))
            flattened_dim = out.view(1, -1).shape[1]

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_dim, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, x):
        features = self.pool(self.feature_extractor(x))
        return self.regression(features)
