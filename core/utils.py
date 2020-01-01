import numpy as np
import json
import time
import os
from densevid_eval.evaluate import ANETcaptions


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
    evaluator = ANETcaptions(ground_truth_filenames=references_path,
                            prediction_filename=candidate_path,
                            tious=[0.3, 0.5, 0.7, 0.9],
                            max_proposals=1000,
                            verbose=False)
    evaluator.evaluate()

    # Output the results
    # if args.verbose:
    #     for i, tiou in enumerate(args.tious):
    #         print '-' * 80
    #         print "tIoU: " , tiou
    #         print '-' * 80
    #         for metric in evaluator.scores:
    #             score = evaluator.scores[metric][i]
    #             print '| %s: %2.4f'%(metric, 100*score)

    # Print the averages
    # print('-' * 80)
    # print('Average across all tIoUs')
    # print('-' * 80)
    # for metric in evaluator.scores:
    #     score = evaluator.scores[metric]
    #     print('| %s: %2.4f' % (metric, 100 * sum(score) / float(len(score))))
