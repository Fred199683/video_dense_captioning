import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from tensorboardX import SummaryWriter

import numpy as np
import os

from .utils import save_json, evaluate, decode_captions
from .dataset import CocoCaptionDataset
from .beam_decoder import BeamSearchDecoder

from .model import EventRNN, CaptionRNN

from config import config as cfg


def pack_collate_fn(batch):
    batch_features, batch_cap_vecs = zip(*batch)
    # batch_features : batch, num_events, event_length, feature_dim
    # cap_vecs : batch, num_events, caption_length

    batch_size, feature_dim = len(batch_features), len(batch_features[0][0][0])
    caption_length = len(batch_cap_vecs[0][0])

    # event_nums : batch
    event_nums = torch.tensor([len(event_features) for event_features in batch_features])
    max_event_num = torch.max(event_nums).item()

    padded_batch_cap_vecs = np.zeros((batch_size, max_event_num, caption_length))
    for i, event_cap_vecs in enumerate(batch_cap_vecs):
        for j, cap_vecs in enumerate(event_cap_vecs):
            padded_batch_cap_vecs[i][j] = cap_vecs

    padded_batch_event_features = np.zeros((batch_size, max_event_num, feature_dim))
    for i, event_features in enumerate(batch_features):
        for j, features in enumerate(event_features):
            padded_batch_event_features[i][j] = np.mean(features, axis=0)

    # event_lens : batch, max_event_num
    event_lens = torch.tensor([[len(features) for features in event_features] + [0] * (max_event_num - len(event_features)) for event_features in batch_features])
    max_event_len = torch.max(event_lens).item()

    padded_batch_caption_features = np.zeros((batch_size, max_event_num, max_event_len, feature_dim))
    for i, event_features in enumerate(batch_features):
        for j, features in enumerate(event_features):
            padded_batch_caption_features[i][j][:len(features)] = features

    batch_mask = torch.arange(max_event_num)[None, :] < event_nums[:, None]
    event_masks = torch.arange(max_event_len)[None, None, :] < event_lens[:, :, None]

    padded_batch_caption_features = torch.from_numpy(padded_batch_caption_features)
    padded_batch_event_features = torch.from_numpy(padded_batch_event_features)
    padded_batch_cap_vecs = torch.from_numpy(padded_batch_cap_vecs)

    return padded_batch_caption_features, padded_batch_event_features, padded_batch_cap_vecs, batch_mask, event_masks


class CaptioningSolver(object):
    def __init__(self, word_to_idx, train_dataset=None, val_dataset=None, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
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

        self.batch_size = cfg.SOLVER.TRAIN.BATCH_SIZE
        self.update_rule = cfg.SOLVER.TRAIN.OPTIM
        self.learning_rate = cfg.SOLVER.TRAIN.LR
        self.n_epochs = cfg.SOLVER.TRAIN.N_EPOCHS
        self.alpha_c = kwargs.pop('alpha_c', 1.0)
        self.eval_every = cfg.SOLVER.TRAIN.EVAL_STEPS
        self.log_path = cfg.SOLVER.TRAIN.LOG_DIR
        self.checkpoint_dir = cfg.SOLVER.TRAIN.CKPT_DIR
        self.checkpoint = cfg.SOLVER.TRAIN.CKPT
        self.results_path = cfg.SOLVER.INFER.RESULT_PATH
        self.capture_scores = kwargs.pop('capture_scores', ['bleu_1', 'bleu_4', 'meteor', 'cider'])

        self.device = cfg.SOLVER.DEVICE

        self.is_test = cfg.TEST.ENABLED and not (cfg.TRAIN.ENABLED or cfg.VAL.ENABLED)

        self.event_rnn = EventRNN(cfg).to(self.device)
        self.caption_rnn = CaptionRNN(cfg, len(word_to_idx)).to(self.device)

        self.beam_decoder = BeamSearchDecoder(self.caption_rnn, len(self.idx_to_word), self._start, self._end, cfg)

        if self.checkpoint is not None:
            self._load(self.checkpoint, is_test=self.is_test)
        else:
            self.start_iter = 0
            self.init_best_scores = {score_name: 0. for score_name in self.capture_scores}

        if not self.is_test:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=pack_collate_fn)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

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
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, self.testing_end_epoch_handler, self.is_test)

        test_pbar = ProgressBar()
        test_pbar.attach(self.test_engine)

    def _save(self, epoch, iteration, loss, best_scores, prefix='epoch'):
        model_name = 'model_' + prefix + '.pth'

        model_dict = {'epoch': epoch,
                      'iteration': iteration,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': loss}
        for metric, score in best_scores.items():
            model_dict[metric] = score

        print('-' * 40)
        print('Saved ' + model_name)
        print('-' * 40)
        torch.save(model_dict, os.path.join(self.checkpoint_dir, model_name))

    def _load(self, model_path, is_test=False):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not is_test:
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

        if iteration % self.eval_every == 0:
            caption_scores = self.test(self.val_loader, is_validation=True)
            for metric, score in caption_scores.items():
                self.writer.add_scalar(metric, score, iteration)
            for metric, score in engine.state.best_scores.items():
                if score < caption_scores[metric]:
                    engine.state.best_scores[metric] = caption_scores[metric]
                    self._save(epoch, iteration, loss, engine.state.best_scores, prefix='best_' + metric)

    def training_end_epoch_handler(self, engine):
        iteration = engine.state.iteration
        epoch = engine.state.epoch
        loss, acc = engine.state.output

        print('-' * 40)
        print('Complete Epoch: {}, Loss:{}, Accuracy:{}'.format(epoch, loss, acc))
        print('-' * 40)
        self.writer.add_scalar('Loss', loss, iteration)
        self.writer.add_scalar('Accuracy', acc, iteration)

        caption_scores = self.test(is_validation=True)
        for metric, score in caption_scores.items():
            self.writer.add_scalar(metric, score, iteration)
        for metric, score in engine.state.best_scores.items():
            if score < caption_scores[metric]:
                engine.state.best_scores[metric] = caption_scores[metric]

        self._save(epoch, iteration, engine.state.output[0], engine.state.best_scores)

    def _train(self, engine, batch):
        self.event_rnn.train()
        self.caption_rnn.train()
        self.optimizer.zero_grad()

        caption_features, event_features, cap_vecs, event_mask, caption_mask = batch
        caption_features = caption_features.to(device=self.device)
        event_features = event_features.to(device=self.device)
        event_mask = event_mask.to(device=self.device)
        caption_mask = caption_mask.to(device=self.device)
        cap_vecs = cap_vecs.to(device=self.device)

        caption_features = self.caption_rnn.normalize(caption_features)
        caption_features_proj = self.caption_rnn.project_features(caption_features)

        event_features = self.event_rnn.normalize(event_features)
        event_features_proj = self.event_rnn.project_features(event_features)

        e_hidden_states, e_cell_states = self.event_rnn.get_initial_lstm(event_features_proj)
        c_hidden_states = self.caption_rnn.zero_hidden_states(batch_size=event_features.size(0))

        losses, accs = 0, 0

        for event_idx in range(event_features.size(1)):
            e_hidden_states, e_cell_states = self.event_rnn(event_idx, event_features, event_features_proj, event_mask,
                                                            e_hidden_states, e_cell_states, c_hidden_states)
            c_hidden_states, c_cell_states = self.caption_rnn.get_initial_lstm(caption_features[:, event_idx])

            feats_alphas = []
            loss, acc = 0, 0
            for caption_idx in range(len(cap_vecs.size(2))):
                curr_cap_vecs = cap_vecs[:, event_idx, caption_idx]

                logits, feats_alpha, (c_hidden_states, c_cell_states) = self.caption_rnn(caption_features[:, event_idx], caption_features_proj[:, event_idx], caption_mask,
                                                                                         curr_cap_vecs, c_hidden_states, c_cell_states)
                loss += self.word_criterion(logits, cap_vecs)
                # acc += torch.sum(torch.argmax(logits, dim=-1)[:caption_batch_sizes[caption_idx+1]] == cap_vecs[end_idx:end_idx+caption_batch_sizes[caption_idx+1]])
                feats_alphas.append(feats_alpha)

            # if self.alpha_c > 0:
            #    sum_loc_alphas = torch.sum(nn.utils.rnn.pad_sequence(feats_alphas), 1)
            #    feats_alphas_reg = self.alpha_c * self.alpha_criterion(sum_loc_alphas, (seq_lens / self.model.L).repeat(1, self.model.L))
            #    loss += feats_alphas_reg

            loss /= caption_features.size(0)
            losses += loss

        losses.backward()
        self.optimizer.step()

        # return loss.item(), float(acc.item()) / float(torch.sum(batch_sizes[1:]).item())
        return loss.item(), accs

    def testing_start_epoch_handler(self, engine):
        engine.state.captions = []

    def testing_end_epoch_handler(self, engine, is_test):
        save_json(engine.state.captions, self.results_path)
        if not is_test:
            print('-' * 40)
            caption_scores = evaluate(candidate_path=self.results_path, get_scores=True)
            for metric, score in caption_scores.items():
                print(metric, ': ', score)
            print('-' * 40)
            engine.state.scores = caption_scores

    def _test(self, engine, batch):
        self.model.eval()
        features, video_ids = batch
        cap_vecs = self.beam_decoder.decode(features)
        captions = decode_captions(cap_vecs.cpu().numpy(), self.idx_to_word)
        video_ids = video_ids.numpy()
        engine.state.captions = engine.state.captions + [{'video_id': int(video_id), 'caption': caption} for video_id, caption in zip(video_ids, captions)]

    def train(self):
        self.train_engine.run(self.train_loader, max_epochs=self.n_epochs)

    def test(self, test_dataset=None, is_validation=False):
        if is_validation is True:
            test_state = self.test_engine.run(self.val_loader)
            return test_state.scores
        else:
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4)
            self.test_engine.run(self.test_loader)
