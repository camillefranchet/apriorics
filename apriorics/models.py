from typing import Any, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from nptyping import NDArray
from torch import nn

from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, DetrFeatureExtractor, DetrForObjectDetection, DeformableDetrForObjectDetection
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datasets import get_coco_api_from_dataset

from apriorics.model_components.axialnet import (
    AxialBlock,
    AxialBlock_dynamic,
    AxialBlock_wopos,
)
from apriorics.model_components.convolution import (
    ConvBnRelu,
    LastCross,
    SelfAttentionBlock,
    conv1x1,
)
from apriorics.model_components.decoder_blocks import DecoderBlock, PixelShuffleICNR
from apriorics.model_components.hooks import Hooks
from apriorics.model_components.normalization import bc_norm, group_norm
from apriorics.model_components.utils import get_sizes
from datasets.coco_eval import CocoEvaluator
from datasets import get_coco_api_from_dataset

class CBR(nn.Module):
    """"""

    def __init__(
        self,
        kernel_size: int,
        n_kernels: int,
        n_layers: int,
        n_classes: int = 2,
        in_chans: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        in_c = in_chans
        out_c = n_kernels
        for k in range(n_layers):
            self.add_module(
                f"cbr{k}",
                ConvBnRelu(
                    in_c,
                    out_c,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    norm_layer=norm_layer,
                ),
            )
            # self.add_module(f'maxpool{k}', nn.MaxPool2d(3, stride=2, padding=1))
            in_c = out_c
            out_c *= 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(out_c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        for m in self.children():
            x = m(x)
        return x


class SASA(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        n_kernels: int,
        n_layers: int,
        n_groups: int,
        n_classes: int = 2,
        in_chans: int = 3,
    ):
        super().__init__()

        self.stem = ConvBnRelu(
            in_chans, n_kernels, 7, stride=2, padding=3, padding_mode="reflect"
        )
        in_c = n_kernels
        out_c = 2 * n_kernels
        for k in range(n_layers):
            self.add_module(
                f"sasa_block_{k}",
                SelfAttentionBlock(in_c, out_c, kernel_size, groups=n_groups),
            )
            self.add_module(f"pool_{k}", nn.AvgPool2d(2, stride=2))
            in_c = out_c
            out_c *= 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten(-2, -1)
        self.fc = nn.Linear(out_c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.children():
            x = m(x)
        return x


class DynamicUnet(nn.Module):
    """"""

    def __init__(
        self,
        encoder_name: str,
        num_classes: int = 2,
        img_size: int = 224,
        img_chan: int = 3,
        pretrained: bool = True,
        norm_layer: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if "cbr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=nn.BatchNorm2d)
            norm_layer = nn.BatchNorm2d
            cut = -3
        elif "cgr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=group_norm)
            cut = -3
        elif "bcr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=bc_norm)
            norm_layer = bc_norm
            cut = -3
        elif "sasa" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = SASA(*args)
            cut = -3
        elif "sanet" in encoder_name:
            splits = encoder_name.split("_")
            kernel_size = int(splits[-1])
            encoder = globals()[splits[0]](kernel_size)
            cut = -2
        else:
            encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                norm_layer=norm_layer,
                features_only=True,
            )
            cut = None

        self.encoder = nn.Sequential(*(list(encoder.children())[:cut] + [nn.ReLU()]))
        encoder_sizes, idxs = self._register_output_hooks(
            input_shape=(img_chan, img_size, img_size)
        )
        n_chans = int(encoder_sizes[-1][1])
        middle_conv = nn.Sequential(
            ConvBnRelu(n_chans, n_chans // 2, 3, norm_layer=norm_layer),
            ConvBnRelu(n_chans // 2, n_chans, 3, norm_layer=norm_layer),
        )
        decoder = [middle_conv]
        for k, (idx, hook) in enumerate(zip(idxs[::-1], self.hooks)):
            skip_chans = int(encoder_sizes[idx][1])
            final_div = k != len(idxs) - 1
            decoder.append(
                DecoderBlock(
                    n_chans,
                    skip_chans,
                    hook,
                    final_div=final_div,
                    norm_layer=norm_layer,
                )
            )
            n_chans = n_chans // 2 + skip_chans
            n_chans = n_chans if not final_div else skip_chans
        self.decoder = nn.Sequential(*decoder, PixelShuffleICNR(n_chans, n_chans))
        self.head = nn.Sequential(
            nn.Conv2d(n_chans + img_chan, n_chans, 1),
            LastCross(n_chans, norm_layer=norm_layer),
            nn.Conv2d(n_chans, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        y = self.decoder(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, x.shape[-2:], mode="nearest")
        y = torch.cat([x, y], dim=1)
        y = self.head(y)
        return y

    def _register_output_hooks(
        self, input_shape: Tuple[int, int, int] = (3, 224, 224)
    ) -> Tuple[NDArray[(Any, Any), int], NDArray[(Any,), int]]:
        sizes, modules = get_sizes(self.encoder, input_shape=input_shape)
        mods = []
        idxs = np.where(sizes[:-1, -1] != sizes[1:, -1])[0]

        def _hook(model, input, output):
            return output

        for k in idxs[::-1]:
            m = modules[k]
            if "downsample" not in m.name:
                mods.append(m)
        self.hooks = Hooks(mods, _hook)

        return sizes, idxs

    def __del__(self):
        if hasattr(self, "hooks"):
            self.hooks.remove()


class ResAxialAttentionUNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=2,
        groups=8,
        width_per_group=64,
        norm_layer=None,
        s=0.125,
        img_size=128,
        imgchan=3,
        **kwargs,
    ):
        super(ResAxialAttentionUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv2 = nn.Conv2d(
            self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            block, int(128 * s), layers[0], kernel_size=(img_size // 2)
        )
        self.layer2 = self._make_layer(
            block,
            int(256 * s),
            layers[1],
            stride=2,
            kernel_size=(img_size // 2),
        )
        self.layer3 = self._make_layer(
            block,
            int(512 * s),
            layers[2],
            stride=2,
            kernel_size=(img_size // 4),
        )
        self.layer4 = self._make_layer(
            block,
            int(1024 * s),
            layers[3],
            stride=2,
            kernel_size=(img_size // 8),
        )

        # Decoder
        self.decoder1 = nn.Conv2d(
            int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1
        )
        self.decoder2 = nn.Conv2d(
            int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder3 = nn.Conv2d(
            int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder4 = nn.Conv2d(
            int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder5 = nn.Conv2d(
            int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1
        )
        self.adjust = nn.Conv2d(
            int(128 * s), num_classes, kernel_size=1, stride=1, padding=0
        )
        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups=self.groups,
                base_width=self.base_width,
                norm_layer=norm_layer,
                kernel_size=kernel_size,
            )
        )
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                    kernel_size=kernel_size,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # AxialAttention Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = F.relu(
            F.interpolate(self.decoder1(x4), scale_factor=(2, 2), mode="bilinear")
        )
        x = torch.add(x, x4)
        x = F.relu(
            F.interpolate(self.decoder2(x), scale_factor=(2, 2), mode="bilinear")
        )
        x = torch.add(x, x3)
        x = F.relu(
            F.interpolate(self.decoder3(x), scale_factor=(2, 2), mode="bilinear")
        )
        x = torch.add(x, x2)
        x = F.relu(
            F.interpolate(self.decoder4(x), scale_factor=(2, 2), mode="bilinear")
        )
        x = torch.add(x, x1)
        x = F.relu(
            F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode="bilinear")
        )
        x = self.adjust(F.relu(x))
        return x

    def forward(self, x):
        return self._forward_impl(x)


class MedTNet(nn.Module):
    def __init__(
        self,
        block,
        block_2,
        layers,
        num_classes=2,
        groups=8,
        width_per_group=64,
        norm_layer=None,
        s=0.125,
        img_size=128,
        imgchan=3,
        **kwargs,
    ):
        super(MedTNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv2 = nn.Conv2d(
            self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            block, int(128 * s), layers[0], kernel_size=(img_size // 2)
        )
        self.layer2 = self._make_layer(
            block,
            int(256 * s),
            layers[1],
            stride=2,
            kernel_size=(img_size // 2),
        )

        self.decoder4 = nn.Conv2d(
            int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder5 = nn.Conv2d(
            int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1
        )
        self.adjust = nn.Conv2d(
            int(128 * s), num_classes, kernel_size=1, stride=1, padding=0
        )
        self.soft = nn.Softmax(dim=1)

        self.conv1_p = nn.Conv2d(
            imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv2_p = nn.Conv2d(
            self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv3_p = nn.Conv2d(
            128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)

        self.relu_p = nn.ReLU(inplace=True)

        img_size_p = img_size // 4

        self.layer1_p = self._make_layer(
            block_2, int(128 * s), layers[0], kernel_size=(img_size_p // 2)
        )
        self.layer2_p = self._make_layer(
            block_2, int(256 * s), layers[1], stride=2, kernel_size=(img_size_p // 2)
        )
        self.layer3_p = self._make_layer(
            block_2, int(512 * s), layers[2], stride=2, kernel_size=(img_size_p // 4)
        )
        self.layer4_p = self._make_layer(
            block_2, int(1024 * s), layers[3], stride=2, kernel_size=(img_size_p // 8)
        )

        # Decoder
        self.decoder1_p = nn.Conv2d(
            int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1
        )
        self.decoder2_p = nn.Conv2d(
            int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder3_p = nn.Conv2d(
            int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder4_p = nn.Conv2d(
            int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1
        )
        self.decoder5_p = nn.Conv2d(
            int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1
        )

        self.decoderf = nn.Conv2d(
            int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1
        )
        self.adjust_p = nn.Conv2d(
            int(128 * s), num_classes, kernel_size=1, stride=1, padding=0
        )
        self.soft_p = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups=self.groups,
                base_width=self.base_width,
                norm_layer=norm_layer,
                kernel_size=kernel_size,
            )
        )
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                    kernel_size=kernel_size,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        xin = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x = F.relu(
            F.interpolate(self.decoder4(x2), scale_factor=(2, 2), mode="bilinear")
        )
        x = torch.add(x, x1)
        x = F.relu(
            F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode="bilinear")
        )
        x_loc = x.clone()

        for i in range(0, 4):
            for j in range(0, 4):

                x_p = xin[:, :, 32 * i : 32 * (i + 1), 32 * j : 32 * (j + 1)]
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                x_p = self.relu(x_p)

                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                x_p = self.relu(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)
                x_p = self.relu(x_p)

                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)

                x_p = F.relu(
                    F.interpolate(
                        self.decoder1_p(x4_p), scale_factor=(2, 2), mode="bilinear"
                    )
                )
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(
                    F.interpolate(
                        self.decoder2_p(x_p), scale_factor=(2, 2), mode="bilinear"
                    )
                )
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(
                    F.interpolate(
                        self.decoder3_p(x_p), scale_factor=(2, 2), mode="bilinear"
                    )
                )
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(
                    F.interpolate(
                        self.decoder4_p(x_p), scale_factor=(2, 2), mode="bilinear"
                    )
                )
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(
                    F.interpolate(
                        self.decoder5_p(x_p), scale_factor=(2, 2), mode="bilinear"
                    )
                )

                x_loc[:, :, 32 * i : 32 * (i + 1), 32 * j : 32 * (j + 1)] = x_p

        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))

        x = self.adjust(F.relu(x))
        return x

    def forward(self, x):
        return self._forward_impl(x)


def axialunet(**kwargs):
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s=0.125, **kwargs)
    return model


def gated(**kwargs):
    model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s=0.125, **kwargs)
    return model


def med_t(**kwargs):
    model = MedTNet(
        AxialBlock_dynamic, AxialBlock_wopos, [1, 2, 4, 1], s=0.125, **kwargs
    )
    return model


def logo(**kwargs):
    model = MedTNet(AxialBlock, AxialBlock, [1, 2, 4, 1], s=0.125, **kwargs)
    return model


def unet(pretrained=True, encoder_name="cbr_5_32_4", **kwargs):
    model = DynamicUnet(encoder_name, pretrained=pretrained, **kwargs)
    return model

# Nécessaire car le formmatage précédent ne fonctionne plus ...
# TODO : Etudier COCOEval de pycocotools pour voir s'il n'y a pas une méthode plus propre
correspondance_stats = {0:'(AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                     1:'(AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
                     2:'(AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
                     3:'(AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                     4:'(AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                     5:'(AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                     6:'(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                     7:'(AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                     8:'(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                     9:'(AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                     10:'(AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                     11:'(AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'}

class YOLOS(pl.LightningModule):

     def __init__(self, lr, weight_decay, val_dataset):
         super().__init__()
         # replace COCO classification head with custom head
         self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small", 
                                                             num_labels=1,
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.weight_decay = weight_decay
         self.iou_types = ['bbox']
         self.base_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute
         self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths
         self.feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)



     def forward(self, pixel_values):
       outputs = self.model(pixel_values=pixel_values)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict


        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        pixel_values = batch["pixel_values"]


        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        self.coco_evaluator.update(res)

        loss = outputs.loss
        loss_dict = outputs.loss_dict         
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())
        
        return loss
    
     def validation_epoch_end(self, validation_step_outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            # Pourquoi est-ce que ça ne marche plus ? Mystère
            #for k, v in coco_eval.formatted.items():
            #        self.log(k, v)
            for ind, stat in enumerate(coco_eval.stats):
                self.log(correspondance_stats[ind], stat)

        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths


     def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, labels=labels)

        pixel_values = batch["pixel_values"]


        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        self.coco_evaluator.update(res)

     def test_epoch_end(self, test_step_outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            # Pourquoi est-ce que ça ne marche plus ? Mystère
            #for k, v in coco_eval.formatted.items():
            #        self.log(k, v)
            for ind, stat in enumerate(coco_eval.stats):
                self.log(correspondance_stats[ind], stat)

        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths
        

     def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, val_dataset):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.iou_types = ['bbox']
        self.base_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute
        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths

    def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
    def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        # forward pass

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        self.coco_evaluator.update(res)

        loss = outputs.loss
        loss_dict = outputs.loss_dict         
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())
        
        return loss
      
    def validation_epoch_end(self, validation_step_outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        

        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            # Pourquoi est-ce que ça ne marche plus ? Mystère
            #for k, v in coco_eval.formatted.items():
            #        self.log(k, v)
            for ind, stat in enumerate(coco_eval.stats):
                self.log(correspondance_stats[ind], stat)

        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, labels=labels)

        pixel_values = batch["pixel_values"]


        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        self.coco_evaluator.update(res)

    def test_epoch_end(self, test_step_outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            # Pourquoi est-ce que ça ne marche plus ? Mystère
            #for k, v in coco_eval.formatted.items():
            #        self.log(k, v)
            for ind, stat in enumerate(coco_eval.stats):
                self.log(correspondance_stats[ind], stat)

        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths
        

   
    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer     

class DeformableDetr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, val_dataset):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr", 
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.feature_extractor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
        self.iou_types = ['bbox']
        self.base_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute
        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths

    def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
    def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        # forward pass

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        self.coco_evaluator.update(res)

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        loss, loss_dict = self.common_step(batch, batch_idx)         
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())   
        
        return loss
     
    def validation_epoch_end(self, validation_step_outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            # Pourquoi est-ce que ça ne marche plus ? Mystère
            #for k, v in coco_eval.formatted.items():
            #        self.log(k, v)
            for ind, stat in enumerate(coco_eval.stats):
                self.log(correspondance_stats[ind], stat)

        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths
    
   
    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer     




class OldDeformdableDetr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, val_dataset):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr", 
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.iou_types = ['bbox']
        self.base_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute
        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths
        #self.feature_extractor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")



    def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
    def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels) 

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels) 
        # forward pass

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())


        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        #results = self.feature_extractor.post_process_object_detection(outputs, orig_target_sizes) # convert outputs of model to COCO api
        results = self.feature_extractor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        self.coco_evaluator.update(res)

        
        return loss
      
    def validation_epoch_end(self, validation_step_outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        for iou_type, coco_eval in self.coco_evaluator.coco_eval.items():
            for k, v in coco_eval.formatted.items():
                    self.log(k, v)

        self.coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types) # initialize evaluator with ground truths


    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer     