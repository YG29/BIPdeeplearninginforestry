import os
import random
import shutil
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm
import logging
from pathlib import Path
import sys
import json

class DataPreprocessor:
    '''class for preprocessing raw data'''
    def __init__(self, config):
        ''''''
        # set up paths
        self.raw_tile_data_dir = Path(config['raw_tile_data_dir'])
        self.raw_mask_data_dir = Path(config['raw_mask_data_dir'])
        self.processed_data_dir = Path(config['processed_data_dir'])
        self.augmented_data_dir = Path(config['augmented_data_dir'])

        # create if not exist
        for dir_path in [self.processed_data_dir, self.augmented_data_dir]:
            for split in ['train', 'val', 'test']:
                (dir_path / split / 'images').mkdir(parents=True, exist_ok=True)
                (dir_path / split / 'masks').mkdir(parents=True, exist_ok=True)

        # split ratios
        self.train_ratio = config.get('train_ratio', 0.8)
        self.val_ratio = config.get('val_ratio', 0.1)
        self.test_ratio = config.get('test_ratio', 0.1)

        # augmentation
        self.augmentation_config = config.get('augmentation', {})
        self.num_augmentations = self.augmentation_config.get('num_per_image', 3)

        # consider logging and setting random seed later

    def split_raw_data(self):
        '''split the data into train val test and save to processed dir'''
        # image tiles
        image_tiles = [file for file in os.listdir(self.raw_tile_data_dir)
                       if file.endswith('.tif')]

