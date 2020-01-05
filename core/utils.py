import numpy as np
import json
import time
import os
import subprocess


def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def sample_coco_minibatch(data, batch_size):
    data_size = data['n_examples']
    mask = np.random.choice(data_size, batch_size)
    file_names = data['file_name'][mask]
    return mask, file_names


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def evaluate(candidate_path='./data/val/val.candidate.captions.json', references_path=['data/val_1.json', 'data/val_2.json'], get_scores=False):
    python = '/home/nii/anaconda3/envs/py27/bin/python'
    subprocess.call('cd densevid_eval && ' + python + ' evaluate.py -s ' + os.path.abspath(candidate_path), shell=True)
