import os
from glob import glob

from torch.utils.data.dataset import Dataset
import numpy as np
from .utils import load_json


class CocoCaptionDataset(Dataset):
    def __init__(self, caption_file, config, split='train'):
        self.split = split
        self.dataset = load_json(caption_file)
        self.video_ids = list(self.dataset.keys())
        self.feature_path = {'train': config.DATASET.TRAIN.FEATURE_PATH,
                             'val': config.DATASET.VAL.FEATURE_PATH,
                             'test': config.DATASET.TEST.FEATURE_PATH}

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        item = self.dataset[video_id]
        event_paths = glob(os.path.join(self.feature_path[self.split], video_id, '*.npy'))
        event_features = [np.load(event_path) for event_path in event_paths]

        if self.split == 'train':
            cap_vec = item['vectors']
            words = item['sentences']
            return event_features, cap_vec, words
        if self.split != 'train':
            timestamps = item['timestamps']
            return event_features, timestamps, video_id

    def __len__(self, ):
        return len(self.dataset)

if __name__ == '__main__':
    print('hello')
