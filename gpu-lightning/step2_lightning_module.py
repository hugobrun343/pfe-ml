"""Step 2: Define a LightningModule (training + validation)."""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import lightning as L

from models import get_model


class Lit3DClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        warmup_epochs: int = 0,
        scheduler: str | None = None,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name, in_channels=3, num_classes=1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.register_buffer("_val_loss_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_loss_count", torch.tensor(0.0), persistent=False)
        self.register_buffer("_tp", torch.tensor(0.0), persistent=False)
        self.register_buffer("_fp", torch.tensor(0.0), persistent=False)
        self.register_buffer("_fn", torch.tensor(0.0), persistent=False)
        self.register_buffer("_tn", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).view(-1)
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self._val_loss_sum.zero_()
        self._val_loss_count.zero_()
        self._tp.zero_()
        self._fp.zero_()
        self._fn.zero_()
        self._tn.zero_()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).view(-1)
        y = y.view(-1)

        val_loss = self.loss_fn(logits, y)
        self._val_loss_sum += val_loss.detach()
        self._val_loss_count += 1.0

        preds = (torch.sigmoid(logits) >= 0.5).float()
        targets = (y >= 0.5).float()

        self._tp += (preds * targets).sum().detach()
        self._fp += (preds * (1.0 - targets)).sum().detach()
        self._fn += ((1.0 - preds) * targets).sum().detach()
        self._tn += ((1.0 - preds) * (1.0 - targets)).sum().detach()

    def on_validation_epoch_end(self):
        if self._val_loss_count.item() == 0:
            return

        val_loss = self._val_loss_sum / self._val_loss_count

        eps = 1e-8
        f1_pos = (2.0 * self._tp) / (2.0 * self._tp + self._fp + self._fn + eps)
        f1_neg = (2.0 * self._tn) / (2.0 * self._tn + self._fn + self._fp + eps)
        f1_mean = (f1_pos + f1_neg) / 2.0

        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log("val_f1_pos", f1_pos, prog_bar=False, logger=True)
        self.log("val_f1_neg", f1_neg, prog_bar=False, logger=True)
        self.log("val_f1_mean", f1_mean, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # Choose optimizer
        if self.hparams.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:  # default: adam
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # No scheduler requested -> return optimizer only
        if self.hparams.scheduler is None:
            return optimizer

        # Build scheduler(s)
        warmup_epochs = max(self.hparams.warmup_epochs, 0)
        cosine_epochs = max(self.hparams.max_epochs - warmup_epochs, 1)

        if self.hparams.scheduler == "cosine":
            cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)

            if warmup_epochs > 0:
                warmup = LinearLR(
                    optimizer,
                    start_factor=1e-3,     # start at lr * 1e-3
                    end_factor=1.0,        # ramp up to full lr
                    total_iters=warmup_epochs,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = cosine

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        # Unknown scheduler -> just return optimizer
        return optimizer
