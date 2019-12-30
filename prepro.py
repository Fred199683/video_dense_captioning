from collections import Counter
from core.utils import save_json, load_json
from config import config as cfg

from tqdm import tqdm
import h5py
import numpy as np

import os
from shutil import rmtree


def process_captions_data(captions_data, max_length=None):
    all_count, removing_count = 0, 0
    empty_videos = []

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
        if sentences == [] and timestamps == []:
            empty_videos.append(video_id)

    for video_id in empty_videos:
        captions_data.pop(video_id)

    print('Removed %d sentences over %d sentences.' % (removing_count, all_count))
    print('There were %d empty videos and they were removed.' % len(empty_videos))

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


def build_caption_vector(captions_data, word_to_idx, vocab_size=30):
    for video_id, annotation in captions_data.items():
        captions_data[video_id]['vectors'] = []
        for i, sentence in enumerate(annotation['sentences']):
            cap_vec = [word_to_idx['<START>']]
            words = ['<START>']

            for word in sentence.split(' '):  # caption contains only lower-case words
                if word in word_to_idx.keys():
                    cap_vec.append(word_to_idx[word])
                    words.append(word)
                else:
                    cap_vec.append(word_to_idx['<UNK>'])
                    words.append('<UNK>')

            cap_vec.append(word_to_idx['<END>'])
            words.append('<END>')

            while len(cap_vec) < vocab_size + 2:
                cap_vec.append(word_to_idx['<NULL>'])
                words.append('<NULL>')

            captions_data[video_id]['vectors'].append(cap_vec)
            captions_data[video_id]['words'][i] = words

    print('Finished building train caption vectors.')
    return captions_data


def main():
    enabled_phases = [('train', cfg.TRAIN.ENABLED), ('val', cfg.VAL.ENABLED), ('test', cfg.TEST.ENABLED)]
    caption_paths = [cfg.DATASET.TRAIN.CAPTION_PATH, cfg.DATASET.VAL.CAPTION_PATH, '']
    feature_paths = [cfg.DATASET.TRAIN.FEATURE_PATH, cfg.DATASET.VAL.FEATURE_PATH, cfg.DATASET.TEST.FEATURE_PATH]

    for (phase, phase_enabled), caption_path, feature_path in zip(enabled_phases, caption_paths, feature_paths):
        if not phase_enabled:
            continue

        captions_data = load_json(caption_path)

        if phase == 'train':
            captions_data = process_captions_data(captions_data, max_length=cfg.DATASET.SEQUENCE_LENGTH)

            word_to_idx = build_vocab(captions_data, threshold=cfg.DATASET.VOCAB_THRESHOLD, vocab_size=cfg.DATASET.VOCAB_SIZE)
            save_json(word_to_idx, cfg.DATASET.VOCAB_PATH)

            captions_data = build_caption_vector(captions_data, word_to_idx=word_to_idx)

        if os.path.isdir(feature_path):
            rmtree(feature_path)
            os.makedirs(feature_path)

        with h5py.File(cfg.DATASET.RAW_FEATURE_PATH) as f_features:
            max_len, warning_count, total_count = 0, 0, 0
            for video_id in tqdm(captions_data.keys()):
                video_duration = captions_data[video_id]['duration']
                event_timestamps = captions_data[video_id]['timestamps']
                event_sentences = captions_data[video_id]['sentences']

                video_feature = f_features[video_id]['c3d_features'][()]
                feature_size = video_feature.shape[0]
                scale_factor = round(feature_size / video_duration)  # To resample features so that every feature represents roughly a second
                if scale_factor == 0:
                    scale_factor += 1

                video_feature = np.pad(video_feature, ((0, (scale_factor - (feature_size % scale_factor)) % scale_factor), (0, 0)))
                video_feature = np.mean(video_feature.reshape(video_feature.shape[0] // scale_factor, -1, video_feature.shape[1]), axis=1)

                video_feature_path = os.path.join(feature_path, video_id)
                os.makedirs(video_feature_path)

                new_event_timestamps, new_event_sentences = [], []

                for i, (begin_timestamp, end_timestamp) in enumerate(event_timestamps):
                    begin_pivot = round(begin_timestamp / video_duration * feature_size / scale_factor)
                    end_pivot = round(end_timestamp / video_duration * feature_size / scale_factor)

                    if begin_pivot != end_pivot:
                        event_feature = video_feature[begin_pivot: end_pivot, :]
                        np.save(os.path.join(video_feature_path, '%d.npy' % i), event_feature)
                        new_event_timestamps.append(event_timestamps[i])
                        new_event_sentences.append(event_sentences[i])

                    else:
                        warning_count += 1
                        if max_len < end_timestamp - begin_timestamp:
                            max_len = end_timestamp - begin_timestamp
                    total_count += 1

                if phase == 'train':
                    captions_data[video_id]['timestamps'] = new_event_timestamps
                    captions_data[video_id]['sentences'] = new_event_sentences

            print('Max length of short events: %d' % max_len)
            print('There are %d short events in %d events.' % (warning_count, total_count))

        if phase == 'train':
            save_json(captions_data, cfg.DATASET.TRAIN.ENC_CAPTION_PATH)


if __name__ == '__main__':
    main()
