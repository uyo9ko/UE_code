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
    def __init__(self, data_dir, img_size, istrain=True, ispredict=False):
        self.ispredict = ispredict
        if self.ispredict:
            self.image_names = os.listdir(data_dir)
            self.uw_image_path = data_dir
            self.transform = A.Compose(
                transforms=[A.Resize(*img_size), ToTensorV2()],
                additional_targets={"image0": "image"},
            )
        else:
            self.uw_image_path = os.path.join(data_dir, "raw")
            self.gt_img_path = os.path.join(data_dir, "reference")
            self.image_names = os.listdir(self.uw_image_path)
            self.transform = A.Compose(
                transforms=[
                    A.Resize(*img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomCrop(256, 256, p=1),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )
            if not istrain:
                self.transform = A.Compose(
                    transforms=[A.Resize(*img_size), ToTensorV2()],
                    additional_targets={"image0": "image"},
                )

    def __getitem__(self, index):
        image_name = self.image_names[index]
        if self.ispredict:
            uw_image = np.array(Image.open(os.path.join(self.uw_image_path, image_name)))
            underwater_image = self.transform(image=uw_image)["image"]
            underwater_image = underwater_image.float() / 255
            return{"underwater_image": underwater_image}
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
        img_size,
        data_path
    ):
        super().__init__()
        self.dataset = data_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.data_path = data_path

    def setup(self, stage=None):
        self.train_data = ImageDataset(os.path.join(self.data_path, "Train"), img_size=self.img_size)
        self.val_data = ImageDataset(os.path.join(self.data_path, "Test"), istrain=False, img_size=self.img_size)
        self.predict_data = ImageDataset(os.path.join(self.data_path, "Test"), ispredict=True, img_size=self.img_size)

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
