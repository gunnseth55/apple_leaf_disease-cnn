import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import ResNet34_Weights, resnet34


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs, skip=None):
        inputs = functional.interpolate(
            inputs,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        if skip is not None:
            if inputs.shape[-2:] != skip.shape[-2:]:
                inputs = functional.interpolate(
                    inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            inputs = torch.cat((inputs, skip), dim=1)
        return self.block(inputs)


class ResNet34UNet(nn.Module):
    """Binary lesion U-Net with a ResNet34 ImageNet encoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)
        self.decoder0 = DecoderBlock(32, 0, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, inputs):
        stem = self.stem(inputs)
        encoder1 = self.encoder1(self.pool(stem))
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        decoded = self.decoder4(encoder4, encoder3)
        decoded = self.decoder3(decoded, encoder2)
        decoded = self.decoder2(decoded, encoder1)
        decoded = self.decoder1(decoded, stem)
        decoded = self.decoder0(decoded)
        return self.output(decoded)
