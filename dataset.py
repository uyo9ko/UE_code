from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import glob
import os
import pytorch_lightning as pl
import random
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):
    def __init__(self, data_dir, istrain=True):
        if istrain:
            self.uw_image_path = os.path.join(data_dir, "train")
        else:
            self.uw_image_path = os.path.join(data_dir, "test")
        self.gt_img_path = os.path.join(data_dir, "gt")
        self.image_names = os.listdir(self.uw_image_path)
        self.transform = A.Compose(
            transforms=[
                A.Resize(448, 608),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(256, 256, p=1),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )
        if not istrain:
            self.transform = A.Compose(
                transforms=[A.Resize(448, 608), ToTensorV2()],
                additional_targets={"image0": "image"},
            )

    def __getitem__(self, index):
        image_name = self.image_names[index]
        uw_image = np.array(Image.open(os.path.join(self.uw_image_path, image_name)))
        gt_image = np.array(Image.open(os.path.join(self.gt_img_path, image_name)))

        transformed = self.transform(image=uw_image, image0=gt_image)
        underwater_image = transformed["image"]
        gt_image = transformed["image0"]
        underwater_image = underwater_image.float() / 255
        gt_image = gt_image.float() / 255
        return {
            "underwater_image": underwater_image,
            "gt_image": gt_image,
            "image_name": image_name,
        }

    def __len__(self):
        return len(self.image_names)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_set,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.dataset = data_set
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        datadir = ""
        if self.dataset == "nyu":
            datadir = "/mnt/epnfs/zhshen/DE_code_0904/nyu_DATA"
        elif self.dataset == "uieb":
            datadir = "/mnt/epnfs/zhshen/DE_code_0904/UIEB"
        self.train_data = ImageDataset(os.path.join(datadir, "Train"))
        self.val_data = ImageDataset(os.path.join(datadir, "Test"), istrain=False)
        self.predict_data = ImageDataset('./samples', istrain=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
