from collections import Counter
from core.utils import save_json, load_json
from config import config as cfg

from tqdm import tqdm
import h5py
import numpy as np

import os


def process_captions_data(ann_file, max_length=None):
    captions_data = load_json(ann_file)

    removing_count = 0
    all_count = 0
    for video_id, annotation in captions_data.items():
        sentences, timestamps = [], []
        all_count += len(annotation['sentences'])

        for sentence, timestamp in zip(annotation['sentences'], annotation['timestamps']):
            sentence = sentence.replace('.', '').replace(',', '').replace("'", '').replace('"', '')
            sentence = sentence.replace('&', 'and').replace('(', '').replace(')', '').replace('-', ' ')
            sentence = ' '.join(sentence.split())  # replace multiple spaces

            if max_length is not None and len(sentence.split(' ')) < max_length:
                sentences.append(sentence.lower())
                timestamps.append(timestamp)
            else:
                removing_count += 1

        captions_data[video_id]['sentences'] = sentences
        captions_data[video_id]['timestamps'] = timestamps

    print('Removed %d sentences over %d sentences.' % (removing_count, all_count))

    return captions_data


def build_vocab(captions_data, threshold=1, vocab_size=None):
    counter = Counter()
    for annotation in captions_data.values():
        for sentence in annotation['sentences']:
            words = sentence.split(' ')  # caption contrains only lower-case words
            for word in sentence.split(' '):
                counter[word] += 1

    if vocab_size is not None:
        top_n_counter = [w for w, n in counter.most_common(vocab_size)]
        vocab = [word for word in counter if counter[word] >= threshold and word in top_n_counter]
    else:
        vocab = [word for word in counter if counter[word] >= threshold]

    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx = len(word_to_idx)
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    return word_to_idx


def build_caption_vector(captions_data, word_to_idx):
    for video_id, annotation in captions_data.items():
        for i, sentence in enumerate(annotation['sentences']):
            cap_vec = [word_to_idx['<START>']]

            for word in sentence.split(' '):  # caption contains only lower-case words
                if word in word_to_idx.keys():
                    cap_vec.append(word_to_idx[word])
                else:
                    cap_vec.append(word_to_idx['<UNK>'])

            cap_vec.append(word_to_idx['<END>'])

            captions_data[video_id]['sentences'][i] = cap_vec

    print('Finished building train caption vectors.')
    return captions_data


def main():
    enabled_phases = [('train', cfg.TRAIN.ENABLED), ('val', cfg.VAL.ENABLED), ('test', cfg.TEST.ENABLED)]
    idx_paths = [cfg.DATASET.TRAIN.IDS_PATH, cfg.DATASET.VAL.IDS_PATH, cfg.DATASET.TEST.IDS_PATH]
    caption_paths = [cfg.DATASET.TRAIN.CAPTION_PATH, cfg.DATASET.VAL.CAPTION_PATH, '']
    feature_paths = [cfg.DATASET.TRAIN.FEATURE_PATH, cfg.DATASET.VAL.FEATURE_PATH, cfg.DATASET.TEST.FEATURE_PATH]

    for (phase, phase_enabled), ids_path, caption_path, feature_path in zip(enabled_phases, idx_paths, caption_paths, feature_paths):
        if not phase_enabled:
            continue

        if phase == 'train':
            captions_data = process_captions_data(caption_path, max_length=cfg.DATASET.SEQUENCE_LENGTH)

            word_to_idx = build_vocab(captions_data, threshold=cfg.DATASET.VOCAB_THRESHOLD, vocab_size=cfg.DATASET.VOCAB_SIZE)
            save_json(word_to_idx, cfg.DATASET.VOCAB_PATH)

            captions_data = build_caption_vector(captions_data, word_to_idx=word_to_idx)
            save_json(captions_data, cfg.DATASET.TRAIN.ENC_CAPTION_PATH)

        if not os.path.isdir(feature_path):
            os.makedirs(feature_path)

        video_ids = load_json(ids_path)
        with h5py.File(cfg.DATASET.RAW_FEATURE_PATH) as f_features:
            for video_id in video_ids:
                video_feature = f_features[video_id]['c3d_features']

                feature_size = video_feature.shape[0]
                video_duration = captions_data[video_id]['duration']
                event_timestamps = captions_data[video_id]['timestamps']

                video_feature_path = os.path.join(feature_path, video_id)

                assert not os.path.isdir(video_feature_path), 'Feature of a video have already been generated. Please remove all features before generating them again.'
                os.makedirs(video_feature_path)

                for i, (begin_timestamp, end_timestamp) in enumerate(event_timestamps):
                    begin_pivot = begin_timestamp / video_duration * feature_size
                    end_pivot = end_timestamp / video_duration * feature_size

                    event_feature = video_feature[begin_pivot: end_timestamp, :]
                    np.save(os.path.join(video_feature_path, '%d.npy' % i))


if __name__ == '__main__':
    main()
