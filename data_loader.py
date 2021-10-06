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


class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, expected_input_size, img_type='png', file_identifier=''):
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

        # self.pre_process = transforms.Compose([transforms.Resize(self.expected_input_size),
        #                                   transforms.ToTensor(),
        #                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.pre_process = transforms.Compose([transforms.CenterCrop(self.expected_input_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    @property
    def file_list(self):
        return [os.path.basename(f) for f in
                glob.glob(os.path.join(self.root_dir, f'*{self.file_identifier}*.{self.img_type}'))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = pil_loader(os.path.join(self.root_dir, img_name))
        # resize images to expected input size (depends on model) and normalize according to ImageNet normalization

        # make a tensor out of the image and normalize according to ImageNet dataset
        image = self.pre_process(image)

        return image, img_name
