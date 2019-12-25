import numpy as np
import json
import time
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


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


def evaluate(candidate_path='./data/val/val.candidate.captions.json', reference_path='./data/val/captions_val2017.json', get_scores=False):
    # load caption data
    ref = COCO(reference_path)
    hypo = ref.loadRes(candidate_path)

    cocoEval = COCOEvalCap(ref, hypo)
    cocoEval.evaluate()
    final_scores = {}
    for metric, score in cocoEval.eval.items():
        final_scores[metric.lower()] = score

    if get_scores:
        return final_scores
