from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.cm import rainbow
from pathaia.util.basic import ifnone
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torchmetrics import Metric
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid

from apriorics.losses import get_loss_name
from apriorics.metrics import MetricCollection
from apriorics.model_components.utils import named_leaf_modules


def get_scheduler_func(
    name: str, total_steps: int, lr: float
) -> Callable[[Optimizer], Optional[Dict[str, Union[str, _LRScheduler]]]]:
    r"""
    Get a function that given an optimizer, returns the corresponding scheduler dict
    formatted for `PytorchLightning <https://www.pytorchlightning.ai/>`_.

    Args:
        name: name of the scheduler. Can either be "one-cycle", "cosine-anneal",
            "reduce-on-plateau" or "none".
        total_steps: total number of training iterations, only useful for "one-cycle" or
            "cosine-anneal".
        lr: baseline learning rate.

    Returns:
        Function that takes an optimizer as input and returns a scheduler dict formatted
        for PytorchLightning.
    """

    def scheduler_func(opt: Optimizer):
        if name == "one-cycle":
            sched = {
                "scheduler": OneCycleLR(opt, lr, total_steps=total_steps),
                "interval": "step",
            }
        elif name == "cosine-anneal":
            sched = {
                "scheduler": CosineAnnealingLR(opt, total_steps),
                "interval": "step",
            }
        elif name == "reduce-on-plateau":
            sched = {
                "scheduler": ReduceLROnPlateau(opt, patience=3),
                "interval": "epoch",
                "monitor": "val_loss",
            }
        else:
            return None
        return sched

    return scheduler_func


class BasicSegmentationModule(pl.LightningModule):
    """
    :class:`pytorch_lightning.LightningModule` to use for binary semantic segmentation
    tasks.

    Args:
        model: underlying PyTorch model.
        loss: loss function.
        lr: learning rate.
        wd: weight decay for AdamW optimizer.
        scheduler_func: Function that takes an optimizer as input and returns a
            scheduler dict formatted for PytorchLightning.
        metrics: list of :class:`torchmetrics.Metric` metrics to compute on validation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        lr: float,
        wd: float,
        scheduler_func: Optional[Callable] = None,
        metrics: Optional[Sequence[Metric]] = None,
        stain_augmentor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.wd = wd
        self.scheduler_func = scheduler_func
        self.metrics = MetricCollection(ifnone(metrics, []))
        self.stain_augmentor = stain_augmentor

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(1)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        if self.stain_augmentor is not None:
            with torch.autocast("cuda", enabled=False):
                x = self.stain_augmentor(x)
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        if batch_idx % 500 == 0 and self.trainer.training_type_plugin.global_rank == 0:
            y_hat = torch.sigmoid(y_hat)
            self.log_images(x[:8], y[:8], y_hat[:8], batch_idx, step="train")

        self.log(f"train_loss_{get_loss_name(self.loss)}", loss)
        if self.sched is not None:
            self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0])
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(f"val_loss_{get_loss_name(self.loss)}", loss, sync_dist=True)

        y_hat = torch.sigmoid(y_hat)
        if batch_idx % 200 == 0 and self.trainer.training_type_plugin.global_rank == 0:
            self.log_images(x[:8], y[:8], y_hat[:8], batch_idx, step="val")

        self.metrics(y_hat, y.int())

    def validation_epoch_end(self, outputs: Dict[str, Tensor]):
        self.log_dict(self.metrics.compute(), sync_dist=True)
        if "SegmentationAUC" in self.metrics:
            met = self.metrics["SegmentationAUC"]
            rec = (met.tp / (met.tp + met.fn + 1e-7)).cpu()
            fpr = (met.fp / (met.tn + met.fp + 1e-7)).cpu()
            prec = (met.tp / (met.tp + met.fp + 1e-7)).cpu()
            self.logger.experiment.log_curve(
                f"ROC_{self.current_epoch}",
                x=fpr.tolist(),
                y=rec.tolist(),
                step=self.current_epoch,
                overwrite=True,
            )
            self.logger.experiment.log_curve(
                f"PRC_{self.current_epoch}",
                x=rec.tolist(),
                y=prec.tolist(),
                step=self.current_epoch,
                overwrite=True,
            )
        self.metrics.reset()

    def configure_optimizers(
        self,
    ) -> Union[
        Optimizer, Dict[str, Union[Optimizer, Dict[str, Union[str, _LRScheduler]]]]
    ]:
        self.opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler_func is None:
            self.sched = None
            return self.opt
        else:
            self.sched = self.scheduler_func(self.opt)
            return {"optimizer": self.opt, "lr_scheduler": self.sched}

    def log_images(
        self, x: Tensor, y: Tensor, y_hat: Tensor, batch_idx: int, step="val"
    ):
        y = y[:, None].repeat(1, 3, 1, 1)
        y_hat = y_hat[:, None].repeat(1, 3, 1, 1)
        sample_imgs = torch.cat((x, y, y_hat))
        grid = make_grid(sample_imgs, y.shape[0])
        self.logger.experiment.log_image(
            to_pil_image(grid),
            f"{step}_image_sample_{self.current_epoch}_{batch_idx}",
            step=self.current_epoch,
        )

    def freeze_encoder(self):
        for m in named_leaf_modules(self.model):
            if "encoder" in m.name and not isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = False


class BasicDetectionModule(pl.LightningModule):
    """
    :class:`PytorchLightning.LightningModule` to use for binary semantic segmentation
    tasks.

    Args:
        model: underlying PyTorch model.
        lr: learning rate.
        wd: weight decay for AdamW optimizer.
        scheduler_func: Function that takes an optimizer as input and returns a
            scheduler dict formatted for PytorchLightning.
        seg_metrics: list of :class:`torchmetrics.Metric` segmentation metrics to
            compute on validation.
        det_metrics: list of :class:`torchmetrics.Metric` detection metrics to
            compute on validation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        wd: float,
        scheduler_func: Optional[Callable] = None,
        seg_metrics: Optional[Sequence[Metric]] = None,
        det_metrics: Optional[Sequence[Metric]] = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.scheduler_func = scheduler_func
        self.seg_metrics = MetricCollection(ifnone(seg_metrics, []))
        self.det_metrics = MetricCollection(ifnone(det_metrics, []))

    def forward(
        self, x: List[Tensor], y: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        return self.model(x, y)

    def training_step(
        self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int
    ) -> Tensor:
        x, y = batch
        losses = self(x, y)
        losses.pop("loss_classifier")

        self.log_dict(losses)
        if self.sched is not None:
            self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0])
        return sum(losses.values())

    def validation_step(
        self,
        batch: Tuple[List[Tensor], List[Dict[str, Tensor]]],
        batch_idx: int,
        *args,
        **kwargs,
    ):
        x, y = batch
        y_hat = self(x)

        if batch_idx % 100 == 0 and self.trainer.training_type_plugin.global_rank == 0:
            self.log_images(x, y, y_hat, batch_idx)

        masks_pred = []
        masks_gt = []
        for gt, pred in zip(y, y_hat):
            if pred["masks"].shape[0]:
                masks_pred.append(
                    (pred["masks"].squeeze(1) * pred["scores"][:, None, None]).amax(0)
                )
            else:
                masks_pred.append(
                    torch.zeros(
                        x.shape[-2:],
                        dtype=pred["masks"].dtype,
                        device=pred["masks"].device,
                    )
                )
            masks_gt.append(gt["masks"].amax(0))
        masks_pred = torch.stack(masks_pred)
        masks_gt = torch.stack(masks_gt)

        self.seg_metrics(masks_pred, masks_gt)
        self.det_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: Dict[str, Tensor]):
        self.log_dict(self.seg_metrics.compute(), sync_dist=True)
        det_dict = self.det_metrics.compute()
        self.seg_metrics.reset()
        self.det_metrics.reset()
        det_dict = {
            k: v
            for k, v in det_dict.items()
            if k.split("_")[-1] not in ("small", "medium", "large", "class")
        }
        self.log_dict(det_dict, sync_dist=True)

    def configure_optimizers(
        self,
    ) -> Union[
        Optimizer, Dict[str, Union[Optimizer, Dict[str, Union[str, _LRScheduler]]]]
    ]:
        self.opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler_func is None:
            self.sched = None
            return self.opt
        else:
            self.sched = self.scheduler_func(self.opt)
            return {"optimizer": self.opt, "lr_scheduler": self.sched}

    def log_images(
        self,
        x: List[Tensor],
        y: List[Dict[str, Tensor]],
        y_hat: List[Dict[str, Tensor]],
        batch_idx: int,
        score_thr: float = 0.7,
    ):
        masks_gt = []
        masks_pred = []
        boxes_gt = []
        boxes_pred = []

        for img, gt, pred in zip(x, y, y_hat):
            img = (img.cpu() * 255).byte()

            idxs = pred["scores"] > score_thr
            cmap = np.linspace(0, 1, idxs.sum())
            if len(cmap):
                colors_preds = [(r, g, b) for r, g, b, _ in rainbow(cmap, bytes=True)]
            else:
                colors_preds = None

            cmap = np.linspace(0, 1, len(gt["masks"]))
            if len(cmap):
                colors_gt = [(r, g, b) for r, g, b, _ in rainbow(cmap, bytes=True)]
            else:
                colors_gt = None

            masks_gt.append(
                draw_segmentation_masks(
                    img, gt["masks"].cpu().bool(), alpha=0.5, colors=colors_gt
                )
            )

            masks = pred["masks"][idxs].cpu().squeeze(1) > 0.5
            masks_pred.append(
                draw_segmentation_masks(img, masks, alpha=0.5, colors=colors_preds)
            )

            boxes_gt.append(
                draw_bounding_boxes(img, gt["boxes"].cpu(), width=2, colors=colors_gt)
            )

            boxes = pred["boxes"][idxs].cpu()
            labels = [f"{score.item(): .2f}" for score in pred["scores"][idxs].cpu()]
            boxes_pred.append(
                draw_bounding_boxes(
                    img, boxes, width=2, colors=colors_preds, labels=labels
                )
            )

        grid = make_grid(masks_gt + masks_pred + boxes_gt + boxes_pred, len(x))
        self.logger.experiment.log_image(
            to_pil_image(grid),
            f"val_image_sample_{self.current_epoch}_{batch_idx}",
            step=self.current_epoch,
        )
