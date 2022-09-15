import re
import numpy as np

import torch
from torch import nn


def dino_vits16(**kwargs):
    vits16 = DinoFeature()
    vits16.expected_input_size = None
    return vits16


class DinoFeature(nn.Module):
    def __init__(
            self,
            dino_arch: str = "vits16",
    ):
        super().__init__()
        self.arch = dino_arch

        if 'vit' in self.arch:
            # patch size of the transformer
            self.patch_size = int(re.findall('[0-9][0-9]|[0-9]', self.arch)[0])
            self.arch = "dino_{}".format(self.arch)
            self.backbone = torch.hub.load('facebookresearch/dino:main', self.arch)
            self.n_feats = self.backbone.norm.bias.shape[0]
        else:
            raise NotImplementedError('Architecture {} not found'.format(self.arch))

    def forward(self, img, n=1):

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        # get selected layer activations
        feat = self.backbone.get_intermediate_layers(x=img, n=n)[0]  # n = 1 means the last layer (gets last n layers)
        # input images/batch size =  B * C * H * W
        # before the ViT: B * C * H*W/patch_size^2 (because flattened)
        # output of ViT: B * * H*W/patch_size^2 + 1 * F (384)
        # (+1 is the cls_token --> aggregation of all patch-level features of ViT (16x16) (NOT input patch)
        # local level representation of all the small patches
        feat_tokens = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
        # global feature aggregation
        feat_cls_token = feat[:, 0, :]
        return feat_cls_token

