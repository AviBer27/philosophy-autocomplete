"""A small Elman next-word model with full-sequence backpropagation.

PyTorch is imported only when this optional module is used. Model files contain
plain JSON weights; loading a trained network never executes a pickle.
"""

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import _tokens, _training_sequences


class _ElmanNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        # BOS has a learned embedding; PAD contributes no embedding gradient.
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim,
                                      padding_idx=vocab_size + 1)
        self.recurrent = nn.RNN(embedding_dim, hidden_size, nonlinearity='tanh',
                                batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden=None):
        states, hidden = self.recurrent(self.embedding(inputs), hidden)
        return self.output(states), hidden


class ElmanLanguageModel:
    """Learn P(x[t] | x[:t]) with an embedding, tanh recurrence and softmax.

    Independent sequences start from a zero hidden state and a distinct learned
    BOS input. Inputs are [BOS, x[0], ..., x[T-2]] and targets are x[:T], including
    EOS. No target or later token is visible to its own prediction. Right padding
    is masked out of both the training loss and reported likelihood. Gradients
    propagate through every valid position of the complete sequence.
    """

    def __init__(self, hidden_size=32, embedding_dim=32, n_epochs=25, seed=7,
                 batch_size=64, learning_rate=.003, patience=5,
                 min_delta=1e-4, verbose=True):
        if hidden_size < 1 or embedding_dim < 1 or n_epochs < 0 or batch_size < 1:
            raise ValueError('Hidden/embedding/batch sizes must be positive; epochs nonnegative.')
        if learning_rate <= 0 or min_delta < 0 or (patience is not None and patience < 1):
            raise ValueError('Invalid learning rate, patience, or minimum improvement.')
        self.hidden_size = int(hidden_size)
        self.embedding_dim = int(embedding_dim)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = None if patience is None else int(patience)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)

    def _initialize(self, vocab_size):
        # CPU avoids device-specific MPS nondeterminism and stays portable.
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.vocab_size = int(vocab_size)
        self.bos_id_ = self.vocab_size
        self.pad_id_ = self.vocab_size + 1
        self.network_ = _ElmanNetwork(vocab_size, self.embedding_dim,
                                       self.hidden_size).cpu()
        self.n_parameters_ = sum(p.numel() for p in self.network_.parameters())

    def _batch(self, sequences):
        lengths = [len(s) for s in sequences]
        inputs = torch.full((len(sequences), max(lengths)), self.pad_id_, dtype=torch.long)
        targets = torch.full_like(inputs, self.pad_id_)
        for row, sequence in enumerate(sequences):
            seq = torch.as_tensor(sequence, dtype=torch.long)
            inputs[row, 0] = self.bos_id_
            if len(seq) > 1:
                inputs[row, 1:len(seq)] = seq[:-1]
            targets[row, :len(seq)] = seq
        return inputs, targets

    def _groups(self, sequences, rng=None):
        # Length grouping bounds padding. Random batch order prevents a fixed
        # short-to-long curriculum while visiting each sequence exactly once.
        ordered = sorted(sequences, key=len)
        groups = [ordered[i:i + self.batch_size]
                  for i in range(0, len(ordered), self.batch_size)]
        if rng is not None:
            rng.shuffle(groups)
        return groups

    def _total_nll(self, sequences):
        total_loss = 0.0
        total_tokens = 0
        self.network_.eval()
        with torch.inference_mode():
            for group in self._groups(sequences):
                inputs, targets = self._batch(group)
                logits, _ = self.network_(inputs)
                total_loss += float(F.cross_entropy(
                    logits.reshape(-1, self.vocab_size).double(), targets.reshape(-1),
                    ignore_index=self.pad_id_, reduction='sum'))
                total_tokens += sum(map(len, group))
        return total_loss, total_tokens

    def corpus_nll(self, sequences):
        seqs = [_tokens(s, self.vocab_size) for s in sequences if len(s)]
        if not seqs:
            raise ValueError('Cannot score an empty corpus.')
        loss, tokens = self._total_nll(seqs)
        return loss / tokens

    def fit(self, sequences, vocab_size, validation_sequences=None):
        seqs = _training_sequences(sequences, vocab_size)
        validation = None if validation_sequences is None else [
            _tokens(s, vocab_size) for s in validation_sequences if len(s)]
        if validation_sequences is not None and not validation:
            raise ValueError('Validation corpus must contain a nonempty sequence.')
        self._initialize(vocab_size)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        self.history_ = []
        self.best_epoch_ = 0
        self.best_validation_nll_ = float('inf')
        best_parameters = None
        significant_best = float('inf')
        stale_epochs = 0
        self.stopped_early_ = False
        fit_start = time.perf_counter()
        for epoch in range(self.n_epochs + 1):
            optimizer_loss = 0.0
            optimizer_tokens = 0
            if epoch:
                self.network_.train()
                for group in self._groups(seqs, rng):
                    inputs, targets = self._batch(group)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = self.network_(inputs)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, self.vocab_size), targets.reshape(-1),
                        ignore_index=self.pad_id_, reduction='sum')
                    tokens = sum(map(len, group))
                    (loss_sum / tokens).backward()
                    # Full BPTT can produce exploding recurrent gradients.
                    nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer_loss += float(loss_sum.detach())
                    optimizer_tokens += tokens
            row = {'epoch': epoch, 'train_nll': self.corpus_nll(seqs)}
            if epoch:
                row['optimizer_train_nll'] = optimizer_loss / optimizer_tokens
            if validation is not None:
                value = self.corpus_nll(validation)
                if not math.isfinite(value):
                    raise FloatingPointError('RNN validation loss became nonfinite.')
                row['validation_nll'] = value
                # Restore the absolute best checkpoint, even if the improvement
                # was too small to reset early stopping's patience counter.
                if value < self.best_validation_nll_:
                    self.best_validation_nll_ = value
                    self.best_epoch_ = epoch
                    best_parameters = {k: v.detach().clone()
                                       for k, v in self.network_.state_dict().items()}
                if value < significant_best - self.min_delta:
                    significant_best, stale_epochs = value, 0
                else:
                    stale_epochs += 1
            row['elapsed_seconds'] = time.perf_counter() - fit_start
            self.history_.append(row)
            if self.verbose:
                val = (f' val_nll={row["validation_nll"]:.5f}'
                       if validation is not None else '')
                print(f'RNN h={self.hidden_size} seed={self.seed} epoch={epoch}'
                      f' train_nll={row["train_nll"]:.5f}{val}', flush=True)
            if (validation is not None and self.patience is not None
                    and stale_epochs >= self.patience and epoch < self.n_epochs):
                self.stopped_early_ = True
                break
        self.epochs_ran_ = self.history_[-1]['epoch']
        self.maxbudget_reached_ = self.epochs_ran_ == self.n_epochs
        if best_parameters is not None:
            self.network_.load_state_dict(best_parameters)
        else:
            self.best_epoch_ = self.epochs_ran_
        self.network_.eval()
        return self

    def next_distribution(self, context_ids=()):
        context = _tokens(context_ids, self.vocab_size)
        inputs = torch.tensor([[self.bos_id_] + context.tolist()], dtype=torch.long)
        self.network_.eval()
        with torch.inference_mode():
            logits, _ = self.network_(inputs)
            return torch.softmax(logits[0, -1].double(), dim=-1).cpu().numpy()

    def iter_distributions(self, sequence):
        seq = _tokens(sequence, self.vocab_size)
        if len(seq):
            for probabilities in next(self.iter_sequence_distributions([seq])):
                yield probabilities

    def iter_sequence_distributions(self, sequences):
        """Yield ordered [length,V] arrays, batching without seeing future words."""
        self.network_.eval()
        pending = []
        for sequence in sequences:
            pending.append(_tokens(sequence, self.vocab_size))
            if len(pending) == self.batch_size:
                yield from self._distribution_batch(pending)
                pending = []
        if pending:
            yield from self._distribution_batch(pending)

    def _distribution_batch(self, sequences):
        nonempty = [s for s in sequences if len(s)]
        if not nonempty:
            for _ in sequences:
                yield np.empty((0, self.vocab_size), dtype=np.float64)
            return
        with torch.inference_mode():
            inputs, _ = self._batch(nonempty)
            logits, _ = self.network_(inputs)
            probabilities = torch.softmax(logits.double(), dim=-1).cpu().numpy()
        row = 0
        for seq in sequences:
            if len(seq):
                yield probabilities[row, :len(seq)].copy()
                row += 1
            else:
                yield np.empty((0, self.vocab_size), dtype=np.float64)

    def sequence_log_likelihood(self, sequence):
        seq = _tokens(sequence, self.vocab_size)
        if not len(seq):
            return 0.0
        loss, _ = self._total_nll([seq])
        return -loss

    def get_state(self):
        return {
            'type': 'elman_rnn', 'hidden_size': self.hidden_size,
            'embedding_dim': self.embedding_dim, 'n_epochs': self.n_epochs,
            'seed': self.seed, 'batch_size': self.batch_size,
            'learning_rate': self.learning_rate, 'patience': self.patience,
            'min_delta': self.min_delta, 'vocab_size': self.vocab_size,
            'weights': {k: v.detach().cpu().tolist()
                        for k, v in self.network_.state_dict().items()},
            'history': self.history_, 'best_epoch': self.best_epoch_,
            'best_validation_nll': (self.best_validation_nll_
                                    if math.isfinite(self.best_validation_nll_) else None),
            'epochs_ran': self.epochs_ran_, 'stopped_early': self.stopped_early_,
            'maxbudget_reached': self.maxbudget_reached_,
        }

    @classmethod
    def from_state(cls, state):
        model = cls(**{key: state[key] for key in (
            'hidden_size', 'embedding_dim', 'n_epochs', 'seed', 'batch_size',
            'learning_rate', 'patience', 'min_delta')}, verbose=False)
        model._initialize(int(state['vocab_size']))
        reference = model.network_.state_dict()
        if set(state['weights']) != set(reference):
            raise ValueError('Saved RNN parameter names do not match the architecture.')
        weights = {}
        for key, value in state['weights'].items():
            tensor = torch.tensor(value, dtype=reference[key].dtype)
            if tensor.shape != reference[key].shape or not torch.isfinite(tensor).all():
                raise ValueError(f'Invalid shape or nonfinite saved RNN weight: {key}')
            weights[key] = tensor
        model.network_.load_state_dict(weights)
        model.network_.eval()
        model.history_ = state.get('history', [])
        model.best_epoch_ = state.get('best_epoch', 0)
        value = state.get('best_validation_nll')
        model.best_validation_nll_ = float('inf') if value is None else float(value)
        model.epochs_ran_ = state.get('epochs_ran', model.best_epoch_)
        model.stopped_early_ = state.get('stopped_early', False)
        model.maxbudget_reached_ = state.get('maxbudget_reached', False)
        return model
