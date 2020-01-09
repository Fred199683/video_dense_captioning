import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from tensorboardX import SummaryWriter

from collections import defaultdict
import numpy as np
import os

from .utils import save_json, load_json, evaluate, decode_captions
from .dataset import CocoCaptionDataset
from .beam_decoder import BeamSearchDecoder

from .model import EventRNN, CaptionRNN

from config import config as cfg


def train_collate(batch):
    batch_features, batch_cap_vecs, batch_sentences = zip(*batch)
    batch_size, feature_dim = len(batch_features), len(batch_features[0][0][0])
    caption_length = len(batch_cap_vecs[0][0])

    # sort batch_features on num_events dimension
    len_sorted_ids = sorted(range(len(batch_features)), key=lambda i: len(batch_features[i]), reverse=True)
    batch_features = [batch_features[i] for i in len_sorted_ids]
    batch_cap_vecs = [batch_cap_vecs[i] for i in len_sorted_ids]
    batch_sentences = [batch_sentences[i] for i in len_sorted_ids]

    event_nums = torch.tensor([len(event_features) for event_features in batch_features])
    max_event_num = torch.max(event_nums).item()

    padded_batch_cap_vecs = torch.zeros(batch_size, max_event_num, caption_length).long()
    for i, event_cap_vecs in enumerate(batch_cap_vecs):
        for j, cap_vecs in enumerate(event_cap_vecs):
            padded_batch_cap_vecs[i][j] = torch.tensor(cap_vecs).long()

    padded_batch_event_features = torch.zeros(batch_size, max_event_num, feature_dim)
    for i, event_features in enumerate(batch_features):
        for j, features in enumerate(event_features):
            mean_feat = torch.mean(torch.tensor(features), axis=0)
            padded_batch_event_features[i][j] = mean_feat

    event_lens = torch.tensor([[len(features) for features in event_features] + [0] * (max_event_num - len(event_features)) for event_features in batch_features])
    max_event_len = torch.max(event_lens).item()

    padded_batch_caption_features = torch.zeros(batch_size, max_event_num, max_event_len, feature_dim)
    for i, event_features in enumerate(batch_features):
        for j, features in enumerate(event_features):
            padded_batch_caption_features[i][j][:len(features)] = torch.tensor(features)

    events_mask = torch.arange(max_event_num)[None, :] < event_nums[:, None]
    captions_masks = torch.arange(max_event_len)[None, None, :] < event_lens[:, :, None]

    batch_sizes = torch.sum(events_mask, dim=0)

    return padded_batch_caption_features, padded_batch_event_features, padded_batch_cap_vecs, events_mask, captions_masks, batch_sizes, batch_sentences


def infer_collate(batch):
    batch_features, batch_timestamps, batch_ids = zip(*batch)

    batch_size, feature_dim = len(batch_features), len(batch_features[0][0][0])

    # sort batch_features on num_events dimension
    len_sorted_ids = sorted(range(len(batch_features)), key=lambda i: len(batch_features[i]), reverse=True)
    batch_ids = [batch_ids[i] for i in len_sorted_ids]
    batch_features = [batch_features[i] for i in len_sorted_ids]
    batch_timestamps = [batch_timestamps[i] for i in len_sorted_ids]

    event_nums = torch.tensor([len(event_features) for event_features in batch_features])
    max_event_num = torch.max(event_nums).item()

    padded_batch_event_features = torch.zeros(batch_size, max_event_num, feature_dim)
    for i, event_features in enumerate(batch_features):
        for j, features in enumerate(event_features):
            padded_batch_event_features[i][j] = torch.mean(torch.tensor(features), axis=0)

    event_lens = torch.tensor([[len(features) for features in event_features] + [0] * (max_event_num - len(event_features)) for event_features in batch_features])
    max_event_len = torch.max(event_lens).item()

    padded_batch_caption_features = torch.zeros(batch_size, max_event_num, max_event_len, feature_dim)
    for i, event_features in enumerate(batch_features):
        for j, features in enumerate(event_features):
            padded_batch_caption_features[i][j][:len(features)] = torch.tensor(features)

    events_mask = torch.arange(max_event_num)[None, :] < event_nums[:, None]
    captions_masks = torch.arange(max_event_len)[None, None, :] < event_lens[:, :, None]

    batch_sizes = torch.sum(events_mask, dim=0)

    return padded_batch_caption_features, padded_batch_event_features, events_mask, captions_masks, batch_sizes, batch_timestamps, batch_ids


class CaptioningSolver(object):
    def __init__(self, word_to_idx, train_dataset=None, val_dataset=None, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
        Optional Arguments:
            - n_epochs: The number of epochs to run for training. - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - snapshot_steps: Integer; training losses will be printed every snapshot_steps iterations.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_checkpoint: String; model path for test
        """

        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.train_batch_size = cfg.SOLVER.TRAIN.BATCH_SIZE
        self.infer_batch_size = cfg.SOLVER.INFER.BATCH_SIZE
        self.update_rule = cfg.SOLVER.TRAIN.OPTIM
        self.learning_rate = cfg.SOLVER.TRAIN.LR
        self.n_epochs = cfg.SOLVER.TRAIN.N_EPOCHS
        self.alpha_c = kwargs.pop('alpha_c', 1.0)
        self.eval_every = cfg.SOLVER.TRAIN.EVAL_STEPS
        self.log_path = cfg.SOLVER.TRAIN.LOG_DIR
        self.checkpoint_dir = cfg.SOLVER.TRAIN.CKPT_DIR
        self.checkpoint = cfg.SOLVER.CHECKPOINT
        self.results_path = cfg.SOLVER.INFER.RESULT_PATH
        self.eval_path = cfg.SOLVER.INFER.EVAL_PATH
        self.capture_scores = cfg.SOLVER.TRAIN.CAPTURED_METRICS

        self.device = cfg.DEVICE

        self.is_train = cfg.TRAIN.ENABLED and cfg.VAL.ENABLED

        self.event_rnn = EventRNN(cfg).to(self.device)
        self.caption_rnn = CaptionRNN(cfg, len(word_to_idx)).to(self.device)

        self.beam_decoder = BeamSearchDecoder(self.caption_rnn, len(self.idx_to_word), self._start, self._end, cfg)

        if self.checkpoint is not None:
            self._load(self.checkpoint, is_train=self.is_train)
        else:
            self.start_iter = 0
            self.init_best_scores = {score_name: 0. for score_name in self.capture_scores}

        if self.is_train:
            self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4, collate_fn=train_collate)
            self.val_loader = DataLoader(val_dataset, batch_size=self.infer_batch_size, num_workers=4, collate_fn=infer_collate)

            # set an optimizer by update rule
            params = list(self.event_rnn.parameters()) + list(self.caption_rnn.parameters())
            if self.update_rule == 'adam':
                self.optimizer = optim.Adam(params=params, lr=self.learning_rate)
            elif self.update_rule == 'rmsprop':
                self.optimizer = optim.RMSprop(params=params, lr=self.learning_rate)

            self.word_criterion = nn.CrossEntropyLoss(ignore_index=self._null, reduction='sum')
            self.alpha_criterion = nn.MSELoss(reduction='sum')

            self.train_engine = Engine(self._train)

            self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, self.training_end_iter_handler)
            self.train_engine.add_event_handler(Events.STARTED, self.training_start_handler)
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, self.training_end_epoch_handler)

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

            self.writer = SummaryWriter(self.log_path, purge_step=self.start_iter)

        self.test_engine = Engine(self._test)

        self.test_engine.add_event_handler(Events.EPOCH_STARTED, self.testing_start_epoch_handler)
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, self.testing_end_epoch_handler, self.is_train)

        test_pbar = ProgressBar()
        test_pbar.attach(self.test_engine)

    def _save(self, epoch, iteration, loss, best_scores, prefix='epoch'):
        model_name = 'model_' + prefix + '.pth'

        model_dict = {'epoch': epoch,
                      'iteration': iteration,
                      'event_state_dict': self.event_rnn.state_dict(),
                      'caption_state_dict': self.caption_rnn.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': loss}
        for metric, score in best_scores.items():
            model_dict[metric] = score

        print('-' * 40)
        print('Saved ' + model_name)
        print('-' * 40)
        torch.save(model_dict, os.path.join(self.checkpoint_dir, model_name))

    def _load(self, model_path, is_train):
        checkpoint = torch.load(model_path)
        self.event_rnn.load_state_dict(checkpoint['event_state_dict'])
        self.caption_rnn.load_state_dict(checkpoint['caption_state_dict'])
        if is_train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_iter = checkpoint['iteration']
        self.init_best_scores = {score_name: checkpoint[score_name] for score_name in self.capture_scores}

        print('-' * 40 + '\nLoaded checkpoint: ' + model_path)
        print('Checkpoint info:\n\tEpoch: %d\n\tIteration: %d' % (checkpoint['epoch'], checkpoint['iteration']))
        print('\t', self.init_best_scores, sep='')
        print('-' * 40)

    def training_start_handler(self, engine):
        engine.state.iteration = self.start_iter
        engine.state.epoch = int(self.start_iter // len(self.train_loader))
        engine.state.best_scores = self.init_best_scores
        print('-' * 40)
        print('Start training at Epoch %d - Iteration %d' % (engine.state.epoch + 1, engine.state.iteration + 1))
        print('Number of iterations per epoch: %d' % len(self.train_loader))
        print('-' * 40)

    def training_end_iter_handler(self, engine):
        iteration = engine.state.iteration
        epoch = engine.state.epoch
        loss, acc = engine.state.output

        print('Epoch: {}, Iteration:{}, Loss:{}, Accuracy:{}'.format(epoch, iteration, loss, acc))
        self.writer.add_scalar('Loss', loss, iteration)
        self.writer.add_scalar('Accuracy', acc, iteration)

        # if iteration % self.eval_every == 0:
        #     caption_scores = self.test(self.val_loader, is_validation=True)
        #     for metric, score in caption_scores.items():
        #         self.writer.add_scalar(metric, score, iteration)
        #     for metric, score in engine.state.best_scores.items():
        #         if score < caption_scores[metric]:
        #             engine.state.best_scores[metric] = caption_scores[metric]
        #             self._save(epoch, iteration, loss, engine.state.best_scores, prefix='best_' + metric)

    def training_end_epoch_handler(self, engine):
        iteration = engine.state.iteration
        epoch = engine.state.epoch
        loss, acc = engine.state.output

        print('-' * 40)
        print('Complete Epoch: {}, Loss:{}, Accuracy:{}'.format(epoch, loss, acc))
        print('-' * 40)
        self.writer.add_scalar('Loss', loss, iteration)
        self.writer.add_scalar('Accuracy', acc, iteration)

        caption_scores = self.test(self.val_loader, is_validation=True)
        for metric, score in caption_scores.items():
            self.writer.add_scalar(metric, score, epoch)
        for metric, score in engine.state.best_scores.items():
            if score < caption_scores[metric]:
                engine.state.best_scores[metric] = caption_scores[metric]
                self._save(epoch, iteration, loss, engine.state.best_scores, prefix='best_' + metric)

        self._save(epoch, iteration, engine.state.output[0], engine.state.best_scores)

    def _train(self, engine, batch):
        self.event_rnn.train()
        self.caption_rnn.train()
        self.optimizer.zero_grad()

        caption_features, event_features, cap_vecs, events_mask, captions_masks, batch_sizes, sentences = batch
        caption_features = caption_features.to(device=self.device)
        event_features = event_features.to(device=self.device)
        events_mask = events_mask.to(device=self.device)
        captions_masks = captions_masks.to(device=self.device)
        batch_sizes = batch_sizes.to(device=self.device)
        cap_vecs = cap_vecs.to(device=self.device)

        caption_features = self.caption_rnn.normalize(caption_features)
        caption_features_proj = self.caption_rnn.project_features(caption_features)

        event_features = self.event_rnn.normalize(event_features)
        event_features_proj = self.event_rnn.project_features(event_features)

        e_hidden_states, e_cell_states = self.event_rnn.get_initial_lstm(event_features_proj)
        c_hidden_states = self.caption_rnn.zero_hidden_states(batch_size=event_features.size(0))

        losses, accs = 0, 0

        sample_captions = []
        for event_idx in range(event_features.size(1)):
            batch_size = batch_sizes[event_idx]
            e_hidden_states, e_cell_states = self.event_rnn(event_idx, event_features[:batch_size], event_features_proj[:batch_size], events_mask[:batch_size],
                                                            e_hidden_states[:, :batch_size], e_cell_states[:, :batch_size], c_hidden_states[:, :batch_size])
            c_hidden_states, c_cell_states = self.caption_rnn.get_initial_lstm(e_hidden_states)

            loss, acc, count_mask = 0., 0., 0.
            feats_alphas, sample_caption = [], []
            captions_mask = captions_masks[:batch_size, event_idx, :]
            for caption_idx in range(cap_vecs.size(2) - 1):
                curr_cap_vecs = cap_vecs[:, event_idx, caption_idx]

                logits, feats_alpha, (c_hidden_states, c_cell_states) = self.caption_rnn(caption_features[:batch_size, event_idx],
                                                                                         caption_features_proj[:batch_size, event_idx], captions_mask,
                                                                                         c_hidden_states[:, :batch_size], c_cell_states[:, :batch_size], curr_cap_vecs[:batch_size])

                next_cap_vecs = cap_vecs[:batch_size, event_idx, caption_idx + 1]
                loss += self.word_criterion(logits, next_cap_vecs)

                mask_next_cap_vecs = (next_cap_vecs != self._null)
                acc += torch.sum((torch.argmax(logits, dim=-1) == next_cap_vecs) * mask_next_cap_vecs).item()
                count_mask += torch.sum(mask_next_cap_vecs).item()
                feats_alphas.append(feats_alpha)

                sample_caption.append(torch.argmax(logits[0]).item())

            if self.alpha_c > 0:
                caption_lens = torch.sum(cap_vecs[:batch_size, event_idx, :] != self._null, dim=-1, keepdim=True).float()
                event_lens = torch.sum(captions_mask, dim=-1, keepdim=True).float()
                sum_loc_alphas = torch.sum(nn.utils.rnn.pad_sequence(feats_alphas), 1)  # N x maxL
                print(sum_loc_alphas.size(), caption_lens.size(), event_lens.size())
                feats_alphas_reg = self.alpha_c * self.alpha_criterion(sum_loc_alphas, (caption_lens / event_lens).repeat(1, sum_loc_alphas.size(-1)))
                loss += feats_alphas_reg

            loss /= caption_features.size(0)
            losses += loss
            accs += acc / count_mask

            sample_captions.append(sample_caption)

        losses.backward()
        self.optimizer.step()
        accs = accs / event_features.size(1)

        print(decode_captions(np.array(sample_captions[0]), self.idx_to_word)[0])
        print(decode_captions(cap_vecs[0][0][1:].cpu().numpy(), self.idx_to_word)[0])

        return loss.item(), accs

    def testing_start_epoch_handler(self, engine):
        engine.state.annotations = {'version': 'VERSION 1.0',
                                    'results': {},
                                    'external_data': {}}

    def testing_end_epoch_handler(self, engine, is_train):
        save_json(engine.state.annotations, self.results_path)
        if is_train:
            print('-' * 40)
            evaluate(candidate_path=self.results_path)
            raw_caption_scores = load_json(self.eval_path)
            caption_scores = {}
            for metric, scores in raw_caption_scores.items():
                score = sum(scores) / float(len(scores))
                print(metric, ': ', score)
                caption_scores[metric.lower()] = score
            print('-' * 40)
            engine.state.scores = caption_scores

    def _test(self, engine, batch):
        self.event_rnn.eval()
        self.caption_rnn.eval()

        caption_features, event_features, events_mask, captions_masks, batch_sizes, timestamps, video_ids = batch
        caption_features = caption_features.to(device=self.device)
        event_features = event_features.to(device=self.device)
        events_mask = events_mask.to(device=self.device)
        captions_masks = captions_masks.to(device=self.device)
        batch_sizes = batch_sizes.to(device=self.device)

        caption_features = self.caption_rnn.normalize(caption_features)
        caption_features_proj = self.caption_rnn.project_features(caption_features)

        event_features = self.event_rnn.normalize(event_features)
        event_features_proj = self.event_rnn.project_features(event_features)

        e_hidden_states, e_cell_states = self.event_rnn.get_initial_lstm(event_features_proj)
        c_hidden_states = self.caption_rnn.zero_hidden_states(batch_size=event_features.size(0))

        predictions = defaultdict(list)
        for event_idx in range(event_features.size(1)):
            batch_size = batch_sizes[event_idx]
            captions_mask = captions_masks[:, event_idx, :]

            e_hidden_states, e_cell_states = self.event_rnn(event_idx, event_features[:batch_size], event_features_proj[:batch_size], events_mask[:batch_size],
                                                            e_hidden_states[:, :batch_size], e_cell_states[:, :batch_size], c_hidden_states[:, :batch_size])
            c_hidden_states, c_cell_states = self.caption_rnn.get_initial_lstm(e_hidden_states)

            cap_vecs = self.beam_decoder.decode(caption_features[:batch_size, event_idx], caption_features_proj[:batch_size, event_idx], captions_mask[:batch_size], c_hidden_states, c_cell_states)

            sentences = decode_captions(cap_vecs.cpu().numpy(), self.idx_to_word)
            for video_id, event_timestamps, sentence in zip(video_ids, timestamps, sentences):
                predictions[video_id].append({'sentence': sentence, 'timestamp': event_timestamps[event_idx]})

        engine.state.annotations['results'].update(predictions)

    def train(self):
        self.train_engine.run(self.train_loader, max_epochs=self.n_epochs)

    def test(self, test_dataset=None, is_validation=False):
        test_state = self.test_engine.run(test_dataset)

        if is_validation is True:
            return test_state.scores
