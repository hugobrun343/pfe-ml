"""Step 2: Define a LightningModule (training + validation)."""

import torch
from torch import nn, optim
import lightning as L

from models import get_model


class Lit3DClassifier(L.LightningModule):
    def __init__(self, model_name: str, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name, in_channels=3, num_classes=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).view(-1)
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).view(-1)
        y = y.view(-1)

        val_loss = self.loss_fn(logits, y)
        self.log("val_loss", val_loss)

        f1_pos, f1_neg = self._f1_per_class(logits, y)
        self.log("val_f1_pos", f1_pos)
        self.log("val_f1_neg", f1_neg)

        f1_mean = (f1_pos + f1_neg) / 2.0
        self.log("val_f1_mean", f1_mean)

    @staticmethod
    def _f1_per_class(logits, targets):
        preds = (torch.sigmoid(logits) >= 0.5).float()
        targets = (targets >= 0.5).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        tn = ((1 - preds) * (1 - targets)).sum()
        fp_neg = fn
        fn_neg = fp

        eps = 1e-8
        f1_pos = (2 * tp) / (2 * tp + fp + fn + eps)
        f1_neg = (2 * tn) / (2 * tn + fp_neg + fn_neg + eps)
        return f1_pos, f1_neg

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
