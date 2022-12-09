"""
Load a dataset of images by specifying the folder where its located.
"""

# Utils
import os
import numpy as np
import glob
from PIL import Image

# Torch related stuff
from torch.utils.data import Dataset
import torch
from torchvision.datasets.folder import pil_loader
from torchvision import transforms

from misc import MyGammaSquare


class ImageDataset(Dataset):

    def __init__(self, root_dir: str, expected_input_size: int, img_type: str = 'png',
                 file_identifier: str = '', normalize: bool = False, crop: int = None,
                 focus_on_centre: bool = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.expected_input_size = expected_input_size
        self.img_type = img_type
        self.file_identifier = file_identifier
        self.crop = crop
        self.normalize = normalize
        self.focus_on_centre = focus_on_centre

    @property
    def pre_process(self):
        pre_process = []
        if self.crop is not None:
            pre_process.append(transforms.CenterCrop(self.crop))

        if self.expected_input_size is not None:
            pre_process.append(transforms.Resize(self.expected_input_size))

        if self.focus_on_centre: # intensify centre, make border lighter
            pre_process.append(MyGammaSquare(0.5, 1.5, 1))

        pre_process.append(transforms.ToTensor())

        if self.normalize:  # ImageNet normalization factors
            pre_process.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))

        return transforms.Compose(pre_process)

    @property
    def file_list(self):
        return sorted([os.path.basename(f) for f in
                glob.glob(os.path.join(self.root_dir, f'*{self.file_identifier}*.{self.img_type}'))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = pil_loader(os.path.join(self.root_dir, img_name))
        # resize images to expected input size (depends on model) and normalize according to ImageNet normalization

        # make a tensor out of the image and normalize according to ImageNet dataset
        image = self.pre_process(image)

        return image, img_name
