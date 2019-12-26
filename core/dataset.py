from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from .utils import load_json

import sys
sys.path.append('..')
from config import config as cfg


class CocoCaptionDataset(Dataset):
    def __init__(self, caption_file, split='train'):
        self.split = split
        self.dataset = load_json(caption_file)
        self.video_ids = self.dataset.keys()
        self.feature_path = {'train': cfg.DATASET.TRAIN.FEATURE_PATH,
                             'val': cfg.DATASET.VAL.FEATURE_PATH,
                             'test': cfg.DATASET.TEST.FEATURE_PATH}

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        item = self.datset[video_id]
        feature_path = os.path.join(self.feature_path[self.split], video_id)
        event_features = [np.load(os.path.join(feature_path, event_feature)) for event_feature in feature_path]

        if self.split == 'train':
            cap_vec = item['sentences']
            return event_features, cap_vec
        return event_features, video_id

    def __len__(self, ):
        return len(self.dataset)

if __name__ == '__main__':
    print('hello')