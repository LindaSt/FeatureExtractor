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

    def __init__(self, root_dir, img_type='png', file_identifier=''):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.img_type = img_type
        self.file_identifier = file_identifier

    @property
    def file_list(self):
        return [os.path.basename(f) for f in glob.glob(os.path.join(self.root_dir, f'*{self.file_identifier}*.{self.img_type}'))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = np.asarray(pil_loader(os.path.join(self.root_dir, img_name)))
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # image = np.transpose(image, (2, 0, 1))
        # TODO: add resize transforamtion to expected input size of network (see how it's done in Gale)
        transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # make a tensor out of the image and normalize according to ImageNet dataset
        image = transformations(image)

        return image, img_name

