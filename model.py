import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from vit_encoder import mit_b4

class AttentionFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(in_channel * 2),
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=int(in_channel / 2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU(),
        )

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channel / 2),
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_a, x_b):
        x = torch.cat((x_a, x_b), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_a * attn[:, 0, :, :].unsqueeze(1) + x_b * attn[:, 1, :, :].unsqueeze(1)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1
        )
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1
        )
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.fusion1 = AttentionFusion(out_channels)
        self.fusion2 = AttentionFusion(out_channels)
        self.fusion3 = AttentionFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

        return out

class Estimation(nn.Module):
    def __init__(self, is_train=False):
        super().__init__()

        self.encoder = mit_b4()
        channels = [512, 320, 128]
        if is_train:
            ckpt_path = "./code/models/weights/mit_b4.pth"
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown

                print("Download pre-trained encoder weights...")
                id = "1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2"
                url = "https://drive.google.com/uc?id=" + id
                output = "./code/models/weights/mit_b4.pth"
                gdown.download(url, output, quiet=False)

        self.decoder_trans = Decoder(channels, 64)
        self.decoder_back = Decoder(channels, 64)

        self.last_layer_trans = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
        self.last_layer_back = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        conv1, conv2, conv3, conv4 = self.encoder(x)
        trans = self.decoder_trans(conv1, conv2, conv3, conv4)
        trans = self.last_layer_trans(trans)
        trans = torch.sigmoid(trans)
        B = self.decoder_back(conv1, conv2, conv3, conv4)
        B = self.last_layer_back(B)
        B = torch.sigmoid(B)
        return trans, B


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimation = Estimation()

    def forward(self, x):
        trans, B = self.estimation(x)
        out = (x - B) / trans

        return trans, B, out
