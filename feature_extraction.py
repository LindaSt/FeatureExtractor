import fire
import os
import pandas as pd
import numpy as np

import torch.nn as nn
from torchvision import models
import torch
from torch.utils.data import DataLoader

from data_loader import ImageDataset


class FeatureExtractor:
    """


    """
    def __init__(self, imgs_path: str, model_name: str, file_identifier: str = '', img_type: str = 'png', batch_size: int = 32):
        self.imgs_path = imgs_path
        self.model_name = model_name
        self.img_type = img_type

        self.batch_size = batch_size
        self.file_identifier = file_identifier

    @property
    def model(self):
        # TODO: change and do like in Gale (copy models from Gale)
        model_fct = eval(f'models.{self.model_name}')
        model = model_fct(pretrained=True)

        # remove last fully-connected layer
        new_classifier = nn.Sequential(*list(model.children())[:-1])
        model.classifier = new_classifier

        return model

    @property
    def data(self):
        return ImageDataset(self.imgs_path, self.img_type, self.file_identifier)

    def get_features(self):
        dataloader = DataLoader(self.data, batch_size=self.batch_size, num_workers=0)
        features = np.array([])
        img_names = np.array([])
        for i_batch, (img_batch, filename_batch) in enumerate(dataloader):
            self.model.eval()
            # Forward Pass
            with torch.no_grad():
                output_batch = self.model(img_batch)
                # save the output features
                img_names = np.append(img_names, list(filename_batch))
                features = np.append(features, np.array(output_batch))

        return features, img_names

def extract_features(imgs_path: str, output_path: str, model: str, file_identifiers: list = [],
                     multifolder: bool = False, img_type: str = 'png'):
    """

    :param file_identifier:
    :param imgs_path:
    :param output_path:
    :param model:
    :param multifolder:
    """

    if not multifolder:
        subfolders = [imgs_path]
    else:
        subfolders = [os.path.join(imgs_path, o) for o in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path, o))]

    feature_table = np.array([])

    for folder in subfolders:
        features = np.array([])
        for file_identifier in file_identifiers:
            feature_extractor = FeatureExtractor(imgs_path=folder, model_name=model, img_type=img_type, file_identifier=file_identifier)
            f = feature_extractor.get_features()

            features = np.append(features, f)

    # create the data frame and save it
    df = pd.DataFrame(feature_table)
    df.to_csv(os.path.join(output_path, f'{model}-patch-features'.png))


if __name__ == '__main__':
    fire.Fire(extract_features)

