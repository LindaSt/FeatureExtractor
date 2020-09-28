import fire
import os
import pandas as pd
import numpy as np
import re
import encoder_models.ResNet as resnet
import encoder_models.VGG as vgg

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from data_loader import ImageDataset


class FeatureExtractor:
    """


    """

    def __init__(self, imgs_path: str, model_name: str, img_type: str = 'png',
                 batch_size: int = 32):
        self.imgs_path = imgs_path
        self.model_name = model_name
        self.img_type = img_type

        self.batch_size = batch_size

    @property
    def model(self):
        # classifier is removed in the forward pass (see file in encoder_models)
        model_category = re.match('\D*', self.model_name).group(0)
        model_fct = eval(f'{model_category}.{self.model_name}')
        model = model_fct(pretrained=True)

        return model

    @property
    def data(self):
        return ImageDataset(root_dir=self.imgs_path, expected_input_size=self.model.expected_input_size, img_type=self.img_type)

    def get_features(self):
        dataloader = DataLoader(self.data, batch_size=self.batch_size, num_workers=0)
        features = {}
        for i_batch, (img_batch, filename_batch) in enumerate(dataloader):
            self.model.eval()
            # Forward Pass
            with torch.no_grad():
                output_batch = self.model(img_batch)
                # save the output features
                assert len(output_batch) == len(filename_batch)
                for fn, output in zip(list(filename_batch), output_batch):
                    features[fn] = np.array(output)

        return features


def extract_features(imgs_path: str, output_path: str, model: str, multifolder: bool = False, img_type: str = 'png'):
    """

    :param imgs_path:
    :param output_path:
    :param model:
    :param multifolder:
    :param img_type:
    """
    if not multifolder:
        subfolders = [imgs_path]
    else:
        subfolders = [os.path.join(imgs_path, o) for o in os.listdir(imgs_path) if
                      os.path.isdir(os.path.join(imgs_path, o))]

    features = {}
    for folder in subfolders:
        feature_extractor = FeatureExtractor(imgs_path=folder, model_name=model, img_type=img_type)
        feature_dict = feature_extractor.get_features()
        features = {**features, **feature_dict}

    # create the data frame and save it
    df = pd.DataFrame.from_dict(features, orient='index')
    df.to_csv(os.path.join(output_path, f'{model}-patch-features.csv'))


if __name__ == '__main__':
    fire.Fire(extract_features)
