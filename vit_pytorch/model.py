import pytorch_lightning as pl
import torch.nn
import torchmetrics
from torch.optim import Adam

from .utils import CosineAnnealingWarmUpRestarts
from .vit import ViT


class LitViT(pl.LightningModule):
    def __init__(self, config, trainer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass')

        self.model = ViT(config)
        self.trainer = trainer

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.forward(batch)
        return loss, accuracy

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.forward(batch)
        return loss, accuracy

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.forward(batch)
        return loss, accuracy

    def setup(self, stage=None):
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self):
        config = self.config
        optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, config.num_epochs)
        return [optimizer], [scheduler]
