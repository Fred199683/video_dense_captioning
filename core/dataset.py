from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from .utils import load_json


class CocoCaptionDataset(Dataset):
    def __init__(self, caption_file, split='train'):
        self.split = split
        dataset = load_json(caption_file)
        if split == 'train':
            self.dataset = dataset['annotations']
            self.word_to_idx = load_json('data/word_to_idx.json')
        else:
            self.dataset = dataset['images']

    def __getitem__(self, index):
        item = self.dataset[index]
        feature_path = os.path.join('data', self.split, 'feats', item['file_name'] + '.npy')
        feature = np.load(feature_path)

        if self.split == 'train':
            caption = item['caption']
            cap_vec = item['vector']
            return feature, cap_vec, caption
        return feature, item['id']

    def __len__(self, ):
        return len(self.dataset)

    def get_vocab_dict(self):
        return self.word_to_idx
