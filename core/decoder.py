import torch
from torch.nn import functional as F
import numpy as np


class Decoder(object):
    def __init__(self, model, vocab_size, start_token, stop_token, cfg):
        self.model = model
        self.vocab_size = vocab_size
        self._start = start_token
        self._end = stop_token
        self.n_time_steps = cfg.SOLVER.INFER.N_TIME_STEPS
        self.beam_size = cfg.SOLVER.INFER.BEAM_SIZE
        self.length_norm = cfg.SOLVER.INFER.LEN_NORM

        self.device = cfg.DEVICE

    def compute_score(self, logits, beam_scores, time_step):
        length_penalty = ((5. + time_step)**self.length_norm) / (6.**self.length_norm)
        score = F.log_softmax(logits, dim=-1) / length_penalty + beam_scores.unsqueeze(-1)
        return score

    def greedy_decode(self, features, features_proj, mask, event_hidden_states, hidden_states, cell_states):
        batch_size, hidden_layers, hidden_size = features.size(0), hidden_states.size(0), hidden_states.size(-1)

        cand_symbols = torch.full([batch_size, self.n_time_steps], self._end).long()
        inputs = torch.full([batch_size, 1], self._start, device=self.device).long()
        for t in range(self.n_time_steps):
            logits, feats_alpha, (hidden_states, cell_states) = self.model(features,
                                                                           features_proj,
                                                                           mask,
                                                                           event_hidden_states,
                                                                           hidden_states,
                                                                           cell_states,
                                                                           inputs)
            inputs = torch.argmax(logits, dim=-1)

        return cand_symbols

    def beam_search_decode(self, features, features_proj, mask, event_hidden_states, hidden_states, cell_states):
        beam_hidden_states = hidden_states.unsqueeze(0)
        beam_cell_states = cell_states.unsqueeze(0)

        batch_size, hidden_layers, hidden_size = features.size(0), hidden_states.size(0), hidden_states.size(-1)

        cand_scores = torch.zeros(batch_size, device=self.device)
        cand_symbols = torch.full([batch_size, self.n_time_steps + 1], self._start, device=self.device).long()
        cand_finished = torch.zeros(batch_size, device=self.device).bool()

        beam_symbols = torch.full([batch_size, 1, 1], self._start, device=self.device).long()
        beam_inputs = torch.full([batch_size, 1], self._start, device=self.device).long()
        beam_scores = torch.zeros(batch_size, 1, device=self.device)

        for t in range(self.n_time_steps):
            beam_size = beam_inputs.size(1)
            beam_logits, next_beam_hidden_states, next_beam_cell_states = [], [], []

            for b in range(beam_size):
                logits, feats_alpha, (hidden_states, cell_states) = self.model(features,
                                                                               features_proj,
                                                                               mask,
                                                                               event_hidden_states,
                                                                               beam_hidden_states[b],
                                                                               beam_cell_states[b],
                                                                               beam_inputs[:, b])
                beam_logits.append(logits.detach())
                next_beam_hidden_states.append(hidden_states.detach())
                next_beam_cell_states.append(cell_states.detach())

            beam_logits = torch.stack(beam_logits, 1)
            beam_hidden_states = torch.stack(next_beam_hidden_states)
            beam_cell_states = torch.stack(next_beam_cell_states)

            symbols_scores = self.compute_score(beam_logits, beam_scores, t)
            end_scores = symbols_scores[:, :, self._end]
            symbols_scores_no_end = torch.cat([symbols_scores[:, :, :self._end],
                                               symbols_scores[:, :, self._end + 1:]], 2).view(batch_size, -1)

            beam_scores, k_indices = torch.topk(symbols_scores_no_end, self.beam_size)

            # Compute immediate candidate
            done_scores_max, done_parent_indices = torch.max(end_scores, -1)
            done_symbols = torch.cat([torch.gather(beam_symbols, 1,
                                      done_parent_indices.view(-1, 1, 1).repeat(1, 1, t + 1)).squeeze(1),
                                      torch.full([batch_size, self.n_time_steps - t],
                                      self._end, dtype=torch.int64, device=self.device)], -1)

            cand_mask = (done_scores_max >= beam_scores[:, -1])
            cand_mask = (cand_mask & ~cand_finished) | ((cand_mask ^ ~cand_finished) & (done_scores_max > cand_scores))
            cand_finished = cand_mask | cand_finished
            cand_symbols = torch.where(cand_mask.unsqueeze(-1), done_symbols, cand_symbols)
            cand_scores = torch.where(cand_mask, done_scores_max, cand_scores)

            # Compute beam candidate for next time-step
            k_symbol_indices = k_indices % (self.vocab_size - 1)
            k_parent_indices = k_indices // (self.vocab_size - 1)
            k_symbol_indices = k_symbol_indices + (k_symbol_indices >= self._end).long()

            past_beam_symbols = torch.gather(beam_symbols, 1,
                                             k_parent_indices.unsqueeze(-1).repeat(1, 1, t + 1))
            beam_symbols = torch.cat([past_beam_symbols, k_symbol_indices.unsqueeze(-1)], -1)

            k_parent_indices = k_parent_indices.t().unsqueeze(1).unsqueeze(-1).repeat(1, hidden_layers, 1, hidden_size)
            beam_hidden_states = torch.gather(beam_hidden_states, 0, k_parent_indices)
            beam_cell_states = torch.gather(beam_cell_states, 0, k_parent_indices)
            beam_inputs = k_symbol_indices
            torch.cuda.empty_cache()

        # if not finished, get the best sequence in beam candidate
        best_beam_symbols = beam_symbols[:, 0]
        cand_symbols = torch.where(cand_finished.unsqueeze(-1), cand_symbols, best_beam_symbols)

        # Remove <START> token
        return cand_symbols[:, 1:]
