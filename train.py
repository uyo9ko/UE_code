import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from torchmetrics.functional import peak_signal_noise_ratio as psnr
from model import *
from dataset import *
from CR import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

save_path = './results/'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1: 
        m.weight.data.normal_(0.0, 0.001)
    if classname.find('Linear') != -1: 
        m.weight.data.normal_(0.0, 0.001)

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MainModel()
        self.model.apply(weights_init)
        self.l1_loss = nn.L1Loss()
        self.CR_loss = ContrastLoss()
        self.save_path = save_path

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        uw_image, gt_image= batch['underwater_image'],batch['gt_image']
        trans_map, atm_map, clear_image = self(uw_image)
        clear_image = torch.clamp(clear_image, 0, 1)
        l1_loss = self.l1_loss(clear_image,gt_image)
        cr_loss = self.CR_loss(clear_image,gt_image,uw_image)
        loss = cr_loss
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        uw_image, gt_image = batch['underwater_image'],batch['gt_image']
        image_name = batch['image_name']
        _trans, _atm, _GT = self(uw_image)
        _GT = torch.clamp(_GT, 0, 1)
        for i in range(len(image_name)):
            # save_image(_trans[i], os.path.join(self.save_path, 'trans_'+image_name[i]))
            # save_image(_atm[i], os.path.join(self.save_path, 'atm_'+image_name[i]))
            save_image(_GT[i], os.path.join(self.save_path, image_name[i]))


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        uw_image, gt_image = batch['underwater_image'],batch['gt_image']
        image_name = batch['image_name']
        _trans, _atm, _GT = self(uw_image)
        _GT = torch.clamp(_GT, 0, 1)
        return _GT
#         for i in range(len(image_name)):
#             # save_image(_trans[i], os.path.join(self.save_path, 'trans_'+image_name[i]))
#             # save_image(_atm[i], os.path.join(self.save_path, 'atm_'+image_name[i]))
#             save_image(_GT[i], os.path.join(self.save_path, image_name[i]))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}


if __name__ == '__main__':
    pl.seed_everything(1234)
    dm = MyDataModule(data_set='uieb',
                            batch_size = 32,
                            num_workers = 64)
    n_epochs = 150
    model = MyModel()
    # model.load_state_dict(torch.load(os.path.join('/mnt/epnfs/zhshen/DE_code_0904/results/mynet_uieb', 'model.pt')))
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=n_epochs,
        check_val_every_n_epoch=10,
    )
    trainer.fit(model, datamodule = dm)
    # predictions = trainer.predict(model, datamodule = dm)
    torch.save(model.state_dict(), os.path.join(save_path,'model.pt'))

    
