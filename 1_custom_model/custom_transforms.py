"""
Provides sets of transforms for training and testing data, respectively. A downscaling factor is provided for reducing
the image resolution.
"""

from torchvision import transforms


class CustomTransforms:
    def __init__(self, downscaling_factor=8):
        self.downscaling_factor = downscaling_factor
        transform_shape = (
            int(420 / downscaling_factor),
            int(1060 / downscaling_factor),
        )

        self.train = transforms.Compose(
            [transforms.Resize(transform_shape), transforms.ToTensor()]
        )

        self.test = transforms.Compose(
            [transforms.Resize(transform_shape), transforms.ToTensor()]
        )

    def __str__(self):
        return f"Transforms_V1, Downscaling factor={self.downscaling_factor}"
