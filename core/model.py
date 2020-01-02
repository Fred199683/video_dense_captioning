# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def mask_softmax(preds, mask, dim=-1):
    preds = preds.masked_fill(~mask, float('-inf'))
    preds = F.softmax(preds, dim=dim)
    preds = preds.masked_fill(~mask, 0)
    return preds


class EventRNN(nn.Module):
    def __init__(self, cfg):
        super(EventRNN, self).__init__()
        self.enable_selector = cfg.MODEL.CRNN.ENABLE_SELECTOR

        self.D = cfg.MODEL.ERNN.D_FEATURE  # size of each region feature
        self.H = cfg.MODEL.ERNN.D_HIDDEN

        # Trainable parameters :
        self.lstm_cell = nn.LSTM(2 * self.D + self.H, self.H, dropout=0.5)
        self.hidden_state_init_layer = nn.Linear(self.D, self.H)
        self.cell_state_init_layer = nn.Linear(self.D, self.H)
        self.feats_proj_layer = nn.Linear(self.D, self.D)
        self.hidden_to_attention_layer = nn.Linear(self.H, self.D)
        self.past_attention_layer = nn.Linear(self.D, 1)
        self.future_attention_layer = nn.Linear(self.D, 1)

        self.features_selector_layer = nn.Linear(self.H, 1)

        # functional layers
        self.features_norm_layer = nn.LayerNorm(self.D)

        self.device = cfg.DEVICE

    def get_initial_lstm(self, feats_proj):
        feats_mean = torch.mean(feats_proj, 1)
        h = torch.tanh(self.hidden_state_init_layer(feats_mean)).unsqueeze(0)
        c = torch.tanh(self.cell_state_init_layer(feats_mean)).unsqueeze(0)
        if torch.sum(torch.isnan(h)) > 0:
            print('-' * 80)
            print('error in event rnn get_initial_lstm.')
            print(feats_proj)
            print(feats_mean)
            print(h)
            print(c)
            print('-' * 80)
        return c, h

    def project_features(self, features):
        batch, loc, dim = features.size()
        features_flat = features.view(-1, dim)
        features_proj = F.relu(self.feats_proj_layer(features_flat))
        features_proj = features_proj.view(batch, loc, -1)
        return features_proj

    def normalize(self, x):
        return self.features_norm_layer(x)

    def _attention_layer(self, features, features_proj, mask, hidden_states, attention_layer):
        h_att = F.relu(features_proj + self.hidden_to_attention_layer(hidden_states[-1]).unsqueeze(1))    # (N, L, D)
        loc, dim = features_proj.size()[1:]
        out_att = attention_layer(h_att.view(-1, dim)).view(-1, loc)   # (N, L)
        alpha = mask_softmax(out_att, mask)
        context = torch.sum(features * alpha.unsqueeze(2), 1)   # (N, D)
        return context, alpha

    def _selector(self, context, hidden_states):
        beta = torch.sigmoid(self.features_selector_layer(hidden_states[-1]))    # (N, 1)
        context = context * beta
        return context, beta

    def forward(self, feature_idx, features, features_proj, mask, hidden_states, cell_states, caption_hidden_states):
        if feature_idx == 0:
            batch_size, _, feature_dim = features.size()
            p_feats_context, p_feats_alpha = torch.zeros(batch_size, feature_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device)
        else:
            p_feats_context, p_feats_alpha = self._attention_layer(features[:, :feature_idx], features_proj[:, :feature_idx], mask[:, :feature_idx],
                                                                   hidden_states, self.past_attention_layer)
        if feature_idx == mask.size(1):
            batch_size, _, feature_dim = features.size()
            f_feats_context, f_feats_alpha = torch.zeros(batch_size, feature_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device)
        else:
            f_feats_context, f_feats_alpha = self._attention_layer(features[:, feature_idx:], features_proj[:, feature_idx:], mask[:, feature_idx:],
                                                                   hidden_states, self.future_attention_layer)

        if self.enable_selector:
            p_feats_context, p_feats_beta = self._selector(p_feats_context, hidden_states)
            f_feats_context, f_feats_beta = self._selector(f_feats_context, hidden_states)

        feats_context = p_feats_context + f_feats_context
        feature = features[:, feature_idx, :]

        next_input = torch.cat((caption_hidden_states.squeeze(0), feats_context, feature), 1).unsqueeze(0)

        output, (next_hidden_states, next_cell_states) = self.lstm_cell(next_input, (hidden_states, cell_states))

        if torch.sum(torch.isnan(next_hidden_states)) > 0:
            print('-' * 80)
            print('error in event rnn at index %d.' % feature_idx)
            if torch.sum(torch.isnan(next_input)) > 0:
                print(torch.sum(torch.isnan(next_input)).item())
                print('\terror in next_input')
                print(next_input)
            if torch.sum(torch.isnan(hidden_states)) > 0:
                print(torch.sum(torch.isnan(hidden_states)).item())
                print('\terror in hidden_states')
                print(hidden_states)
            if torch.sum(torch.isnan(cell_states)) > 0:
                print(torch.sum(torch.isnan(cell_states)).item())
                print('\terror in cell_states')
                print(cell_states)
            print('-' * 80)
            sys.exit()

        return next_hidden_states, next_cell_states


class CaptionRNN(nn.Module):
    def __init__(self, cfg, vocab_length):
        super(CaptionRNN, self).__init__()

        self.prev2out = cfg.MODEL.CRNN.ENABLE_PREV2OUT
        self.ctx2out = cfg.MODEL.CRNN.ENABLE_CTX2OUT
        self.enable_selector = cfg.MODEL.CRNN.ENABLE_SELECTOR
        self.dropout = cfg.MODEL.CRNN.DROPOUT
        self.V = vocab_length
        self.D = cfg.MODEL.CRNN.D_FEATURE  # size of each region feature
        self.M = cfg.MODEL.CRNN.D_EMBED
        self.H = cfg.MODEL.CRNN.D_HIDDEN

        self.device = cfg.DEVICE

        # Trainable parameters :
        self.lstm_cell = nn.LSTM(self.D + self.M, self.H, dropout=0.5)
        self.embedding_lookup = nn.Embedding(self.V, self.M)
        self.feats_proj_layer = nn.Linear(self.D, self.D)
        self.hidden_to_attention_layer = nn.Linear(self.H, self.D)
        self.attention_layer = nn.Linear(self.D, 1)

        self.features_selector_layer = nn.Linear(self.H, 1)

        self.hidden_to_embedding_layer = nn.Linear(self.H, self.M)
        self.features_context_to_embedding_layer = nn.Linear(self.D, self.M)
        self.embedding_to_output_layer = nn.Linear(self.M, self.V)

        # functional layers
        self.features_norm_layer = nn.LayerNorm(self.D)
        self.dropout = nn.Dropout(p=self.dropout)

    def get_initial_lstm(self, event_hidden_states):
        h = event_hidden_states
        c = event_hidden_states
        return c, h

    def zero_hidden_states(self, batch_size):
        return torch.zeros(1, batch_size, self.H, device=self.device)

    def project_features(self, features):
        batch, event_num, event_len, feature_dim = features.size()
        features_flat = features.view(-1, feature_dim)
        features_proj = F.relu(self.feats_proj_layer(features_flat))
        features_proj = features_proj.view(batch, event_num, event_len, -1)
        return features_proj

    def normalize(self, x):
        return self.features_norm_layer(x)

    def word_embedding(self, inputs):
        embed_inputs = self.embedding_lookup(inputs)  # (N, T, M) or (N, M)
        return embed_inputs

    def _attention_layer(self, features, features_proj, mask, hidden_states):
        h_att = F.relu(features_proj + self.hidden_to_attention_layer(hidden_states[-1]).unsqueeze(1))    # (N, L, D)
        loc, dim = features_proj.size()[1:]
        out_att = self.attention_layer(h_att.view(-1, dim)).view(-1, loc)   # (N, L)
        alpha = mask_softmax(out_att, mask)
        context = torch.sum(features * alpha.unsqueeze(2), 1)   # (N, D)
        return context, alpha

    def _selector(self, context, hidden_states):
        beta = torch.sigmoid(self.features_selector_layer(hidden_states[-1]))    # (N, 1)
        context = context * beta
        return context, beta

    def _decode_lstm(self, x, h, feats_context):
        h = self.dropout(h)
        h_logits = self.hidden_to_embedding_layer(h)

        if self.ctx2out:
            h_logits += self.features_context_to_embedding_layer(feats_context)

        if self.prev2out:
            h_logits += x
        h_logits = torch.tanh(h_logits)

        h_logits = self.dropout(h_logits)
        out_logits = self.embedding_to_output_layer(h_logits)
        return out_logits

    def forward(self, features, features_proj, mask, hidden_states, cell_states, past_captions):
        emb_captions = self.word_embedding(inputs=past_captions)

        feats_context, feats_alpha = self._attention_layer(features, features_proj, mask, hidden_states)

        if self.enable_selector:
            feats_context, feats_beta = self._selector(feats_context, hidden_states)

        next_input = torch.cat((emb_captions, feats_context), 1).unsqueeze(0)

        output, (next_hidden_states, next_cell_states) = self.lstm_cell(next_input, (hidden_states, cell_states))

        logits = self._decode_lstm(emb_captions, output.squeeze(0), feats_context)

        return logits, feats_alpha, (next_hidden_states, next_cell_states)
