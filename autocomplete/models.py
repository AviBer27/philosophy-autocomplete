"""Language models implemented with NumPy and the Python standard library.

Every input is an independent text sequence (for example, a radio turn). The caller supplies EOS tokens;
the models never add them. Token 0 is UNK, token 1 is EOS. BOS is an internal
context marker for n-grams and is never a predicted vocabulary entry.
"""

from collections import Counter, defaultdict
from typing import Iterable, Iterator, Optional, Sequence
import time

import numpy as np


def _normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    totals = values.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Cannot normalize an empty probability row.")
    return values / totals


def _tokens(sequence: Sequence[int], vocab_size: int) -> np.ndarray:
    """Map out-of-vocabulary integer IDs to UNK, consistently at inference."""
    values = np.asarray(sequence, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("A sequence must be a one-dimensional list of token IDs.")
    return np.where((values >= 0) & (values < vocab_size), values, 0)


def _training_sequences(sequences, vocab_size):
    if vocab_size < 2:
        raise ValueError("Vocabulary must contain UNK and EOS (at least two IDs).")
    result = [_tokens(s, vocab_size) for s in sequences if len(s)]
    if not result:
        raise ValueError("At least one nonempty training sequence is required.")
    return result


class Unigram:
    """Add-alpha token frequencies, ignoring all preceding context."""

    def __init__(self, alpha: float = 0.1):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)

    def fit(self, sequences, vocab_size: int):
        seqs = _training_sequences(sequences, vocab_size)
        self.vocab_size = int(vocab_size)
        counts = np.full(vocab_size, self.alpha, dtype=np.float64)
        for seq in seqs:
            counts += np.bincount(seq, minlength=vocab_size)
        self.probabilities_ = counts / counts.sum()
        return self

    def next_distribution(self, context_ids=()) -> np.ndarray:
        return self.probabilities_.copy()

    def iter_distributions(self, sequence) -> Iterator[np.ndarray]:
        for _ in sequence:
            yield self.probabilities_.copy()

    def sequence_log_likelihood(self, sequence) -> float:
        seq = _tokens(sequence, self.vocab_size)
        return float(np.log(self.probabilities_[seq]).sum())

    def get_state(self) -> dict:
        return {"type": "unigram", "alpha": self.alpha,
                "vocab_size": self.vocab_size,
                "probabilities": self.probabilities_.copy()}

    @classmethod
    def from_state(cls, state):
        model = cls(state["alpha"])
        model.vocab_size = int(state["vocab_size"])
        model.probabilities_ = np.asarray(state["probabilities"], dtype=np.float64)
        return model


class NGram:
    """Interpolated n-gram model (order 2 or 3).

    A seen context contributes ``interpolation * P_ML(word | context)`` and
    the remaining mass comes from its shorter-context distribution. An unseen
    context backs off completely. The base unigram uses add-alpha smoothing.
    Each sequence starts with order-1 BOS markers, preserving sequence boundaries.
    """

    BOS = -1

    def __init__(self, order: int = 2, alpha: float = 0.1,
                 interpolation: float = 0.8):
        if order not in (2, 3):
            raise ValueError("Only bigram (2) and trigram (3) are supported.")
        if not 0 <= interpolation < 1:
            raise ValueError("interpolation must be in [0,1) for positive support.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.order = int(order)
        self.alpha = float(alpha)
        self.interpolation = float(interpolation)

    def fit(self, sequences, vocab_size: int):
        seqs = _training_sequences(sequences, vocab_size)
        self.vocab_size = int(vocab_size)
        self.unigram_ = Unigram(self.alpha).fit(seqs, vocab_size)
        self.counts_ = {n: defaultdict(Counter) for n in range(1, self.order)}
        for seq in seqs:
            history = [self.BOS] * (self.order - 1)
            for token in seq:
                for n in range(1, self.order):
                    self.counts_[n][tuple(history[-n:])][int(token)] += 1
                history.append(int(token))
        # Store sparse rows; the output vocabulary can be large.
        self.rows_ = {}
        for n, contexts in self.counts_.items():
            self.rows_[n] = {}
            for context, counts in contexts.items():
                ids = np.fromiter(counts.keys(), dtype=np.int64)
                values = np.fromiter(counts.values(), dtype=np.float64)
                self.rows_[n][context] = (ids, values / values.sum())
        return self

    def next_distribution(self, context_ids=()) -> np.ndarray:
        context = _tokens(context_ids, self.vocab_size).tolist()
        history = [self.BOS] * max(0, self.order - 1 - len(context)) + context
        probs = self.unigram_.probabilities_.copy()
        for n in range(1, self.order):
            row = self.rows_[n].get(tuple(history[-n:]))
            if row is not None:
                probs *= 1.0 - self.interpolation
                ids, weights = row
                probs[ids] += self.interpolation * weights
        return probs

    def iter_distributions(self, sequence) -> Iterator[np.ndarray]:
        seq = _tokens(sequence, self.vocab_size)
        context = []
        for token in seq:
            yield self.next_distribution(context)
            context.append(int(token))
            context = context[-(self.order - 1):]

    def sequence_log_likelihood(self, sequence) -> float:
        seq = _tokens(sequence, self.vocab_size)
        return float(sum(np.log(probs[token]) for probs, token in
                         zip(self.iter_distributions(seq), seq)))

    def get_state(self) -> dict:
        rows = [{"n": n, "context": list(context), "ids": ids.copy(),
                 "probabilities": probabilities.copy()}
                for n, contexts in self.rows_.items()
                for context, (ids, probabilities) in contexts.items()]
        return {"type": "ngram", "order": self.order, "alpha": self.alpha,
                "interpolation": self.interpolation, "vocab_size": self.vocab_size,
                "unigram": self.unigram_.get_state(), "rows": rows}

    @classmethod
    def from_state(cls, state):
        model = cls(state["order"], state["alpha"], state["interpolation"])
        model.vocab_size = int(state["vocab_size"])
        model.unigram_ = Unigram.from_state(state["unigram"])
        model.rows_ = {n: {} for n in range(1, model.order)}
        for row in state["rows"]:
            model.rows_[int(row["n"])][tuple(row["context"])] = (
                np.asarray(row["ids"], dtype=np.int64),
                np.asarray(row["probabilities"], dtype=np.float64))
        return model


class CategoricalHMM:
    """Categorical HMM trained by batch Baum-Welch (expectation maximization).

    A[i,j] is P(z[t+1]=j | z[t]=i); B[j,w] is P(x[t]=w | z[t]=j).
    Forward and backward messages are scaled at every timestep. Expected
    sufficient statistics are accumulated in bounded batches of similarly
    sized sequences; EM updates occur only after the entire training corpus.

    ``pseudocount`` is added to each initial, transition and emission count.
    With zero pseudocount, this is maximum-likelihood EM; with a positive value
    it is a smoothed/MAP-style update, so raw training likelihood need not be
    monotone. Validation selects and restores the best epoch, including epoch 0.
    """

    def __init__(self, n_states: int = 8, n_iter: int = 15, seed: int = 42,
                 pseudocount: float = 0.01, batch_size: int = 128,
                 patience=None, min_delta: float = 1e-4, verbose=False):
        if n_states < 1 or n_iter < 0 or batch_size < 1 or pseudocount < 0:
            raise ValueError("Invalid HMM state/iteration/batch/pseudocount value.")
        if min_delta < 0 or (patience is not None and patience < 1):
            raise ValueError("min_delta must be nonnegative and patience positive or None.")
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.pseudocount = float(pseudocount)
        self.batch_size = int(batch_size)
        self.patience = None if patience is None else int(patience)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)

    def _initialize(self, seqs, vocab_size):
        self.vocab_size = int(vocab_size)
        rng = np.random.default_rng(self.seed)
        self.pi_ = _normalize(rng.gamma(1.0, 1.0, self.n_states))
        self.A_ = _normalize(rng.gamma(1.0, 1.0,
                                       (self.n_states, self.n_states)))
        base = np.ones(vocab_size, dtype=np.float64)
        for seq in seqs:
            base += np.bincount(seq, minlength=vocab_size)
        self.B_ = _normalize(rng.gamma(0.7, 1.0,
                                       (self.n_states, vocab_size)) * base)

    def _batches(self, seqs):
        ordered = sorted(seqs, key=len)
        for start in range(0, len(ordered), self.batch_size):
            group = ordered[start:start + self.batch_size]
            lengths = np.array([len(s) for s in group], dtype=np.int64)
            observations = np.zeros((len(group), int(lengths.max())), dtype=np.int64)
            for i, seq in enumerate(group):
                observations[i, :len(seq)] = seq
            yield observations, lengths

    def _forward(self, observations, lengths):
        batch, steps = observations.shape
        alpha = np.zeros((batch, steps, self.n_states), dtype=np.float64)
        scales = np.ones((batch, steps), dtype=np.float64)
        first = self.pi_[None, :] * self.B_[:, observations[:, 0]].T
        scales[:, 0] = first.sum(axis=1)
        if np.any(scales[:, 0] <= 0):
            raise FloatingPointError("The model assigns zero probability to a sequence.")
        alpha[:, 0] = first / scales[:, 0, None]
        for t in range(1, steps):
            active = lengths > t
            current = (alpha[active, t - 1] @ self.A_) * self.B_[:, observations[active, t]].T
            scales[active, t] = current.sum(axis=1)
            if np.any(scales[active, t] <= 0):
                raise FloatingPointError("The model assigns zero probability to a sequence.")
            alpha[active, t] = current / scales[active, t, None]
        return alpha, scales

    def _expectation(self, seqs):
        pi_counts = np.zeros_like(self.pi_)
        a_counts = np.zeros_like(self.A_)
        b_counts = np.zeros_like(self.B_)
        log_likelihood = 0.0
        for obs, lengths in self._batches(seqs):
            alpha, scales = self._forward(obs, lengths)
            batch, steps = obs.shape
            beta = np.zeros_like(alpha)
            beta[np.arange(batch), lengths - 1] = 1.0
            for t in range(steps - 2, -1, -1):
                active = lengths > t + 1
                future = self.B_[:, obs[active, t + 1]].T * beta[active, t + 1]
                beta[active, t] = (future @ self.A_.T) / scales[active, t + 1, None]
                # Sum xi over independent sequences, without constructing B*K*K.
                a_counts += self.A_ * (alpha[active, t].T @
                                        (future / scales[active, t + 1, None]))
            gamma = alpha * beta
            # Normalization absorbs harmless floating-point roundoff.
            mask = np.arange(steps)[None, :] < lengths[:, None]
            gamma[mask] /= gamma[mask].sum(axis=1, keepdims=True)
            pi_counts += gamma[:, 0].sum(axis=0)
            # np.add.at handles repeated word IDs without dropping increments.
            for state in range(self.n_states):
                np.add.at(b_counts[state], obs[mask], gamma[:, :, state][mask])
            log_likelihood += float(np.log(scales).sum())
        return pi_counts, a_counts, b_counts, log_likelihood

    def corpus_nll(self, sequences) -> float:
        seqs = [_tokens(s, self.vocab_size) for s in sequences if len(s)]
        total = sum(map(len, seqs))
        if total == 0:
            raise ValueError("Cannot score an empty corpus.")
        log_likelihood = 0.0
        for obs, lengths in self._batches(seqs):
            _, scales = self._forward(obs, lengths)
            log_likelihood += float(np.log(scales).sum())
        return -log_likelihood / total

    def fit(self, sequences, vocab_size: int, validation_sequences=None):
        seqs = _training_sequences(sequences, vocab_size)
        validation = None if validation_sequences is None else [
            _tokens(s, vocab_size) for s in validation_sequences if len(s)]
        if validation_sequences is not None and not validation:
            raise ValueError("Validation corpus must contain a nonempty sequence.")
        self._initialize(seqs, vocab_size)
        self.history_ = []
        self.best_epoch_ = 0
        self.best_validation_nll_ = float("inf")
        best_parameters = None
        significant_best, stale_epochs = float("inf"), 0
        self.stopped_early_ = False
        tokens = sum(map(len, seqs))
        fit_start = time.perf_counter()
        for epoch in range(self.n_iter + 1):
            # E-step provides both training likelihood and update statistics.
            pi, transitions, emissions, log_likelihood = self._expectation(seqs)
            row = {"epoch": epoch, "train_nll": -log_likelihood / tokens}
            if validation is not None:
                row["validation_nll"] = self.corpus_nll(validation)
                if row["validation_nll"] < self.best_validation_nll_:
                    self.best_validation_nll_ = row["validation_nll"]
                    self.best_epoch_ = epoch
                    best_parameters = (self.pi_.copy(), self.A_.copy(), self.B_.copy())
                if row["validation_nll"] < significant_best - self.min_delta:
                    significant_best, stale_epochs = row["validation_nll"], 0
                else:
                    stale_epochs += 1
            row['elapsed_seconds'] = time.perf_counter() - fit_start
            self.history_.append(row)
            if self.verbose:
                val = (f' val_nll={row["validation_nll"]:.5f}'
                       if validation is not None else '')
                print(f'HMM k={self.n_states} seed={self.seed} epoch={epoch}'
                      f' train_nll={row["train_nll"]:.5f}{val}', flush=True)
            if (validation is not None and self.patience is not None
                    and stale_epochs >= self.patience and epoch < self.n_iter):
                self.stopped_early_ = True
                break
            if epoch == self.n_iter:
                break
            pseudo = self.pseudocount
            self.pi_ = _normalize(pi + pseudo)
            # A state with no outgoing observations can occur with length-1 data.
            transition_rows = transitions + pseudo
            empty = transition_rows.sum(axis=1) == 0
            transition_rows[empty] = self.A_[empty]
            self.A_ = _normalize(transition_rows)
            emission_rows = emissions + pseudo
            empty = emission_rows.sum(axis=1) == 0
            emission_rows[empty] = self.B_[empty]
            self.B_ = _normalize(emission_rows)
        if best_parameters is not None:
            self.pi_, self.A_, self.B_ = best_parameters
        else:
            self.best_epoch_ = self.history_[-1]['epoch']
        self.epochs_ran_ = self.history_[-1]['epoch']
        self.maxbudget_reached_ = self.epochs_ran_ == self.n_iter
        return self

    def _filtered_state(self, context_ids):
        seq = _tokens(context_ids, self.vocab_size)
        if not len(seq):
            return None
        filtered = self.pi_ * self.B_[:, seq[0]]
        total = filtered.sum()
        if total <= 0:
            raise FloatingPointError("Context has zero probability.")
        filtered /= total
        for token in seq[1:]:
            filtered = (filtered @ self.A_) * self.B_[:, token]
            total = filtered.sum()
            if total <= 0:
                raise FloatingPointError("Context has zero probability.")
            filtered /= total
        return filtered

    def next_distribution(self, context_ids=()) -> np.ndarray:
        filtered = self._filtered_state(context_ids)
        state = self.pi_ if filtered is None else filtered @ self.A_
        return state @ self.B_

    def iter_distributions(self, sequence) -> Iterator[np.ndarray]:
        seq = _tokens(sequence, self.vocab_size)
        prior = self.pi_.copy()
        for token in seq:
            yield prior @ self.B_
            posterior = prior * self.B_[:, token]
            total = posterior.sum()
            if total <= 0:
                raise FloatingPointError("Context has zero probability.")
            prior = (posterior / total) @ self.A_

    def sequence_log_likelihood(self, sequence) -> float:
        seq = _tokens(sequence, self.vocab_size)
        if not len(seq):
            return 0.0
        _, scales = self._forward(seq[None, :], np.array([len(seq)]))
        return float(np.log(scales).sum())

    def get_state(self) -> dict:
        return {"type": "hmm", "n_states": self.n_states, "n_iter": self.n_iter,
                "seed": self.seed, "pseudocount": self.pseudocount,
                "batch_size": self.batch_size, "vocab_size": self.vocab_size,
                "patience": self.patience, "min_delta": self.min_delta,
                "pi": self.pi_.copy(), "A": self.A_.copy(), "B": self.B_.copy(),
                "history": self.history_, "best_epoch": self.best_epoch_,
                "epochs_ran": getattr(self, 'epochs_ran_', self.best_epoch_),
                "stopped_early": getattr(self, 'stopped_early_', False),
                "maxbudget_reached": getattr(self, 'maxbudget_reached_', False),
                "best_validation_nll": (self.best_validation_nll_
                                        if np.isfinite(self.best_validation_nll_)
                                        else None)}

    @classmethod
    def from_state(cls, state):
        model = cls(state["n_states"], state["n_iter"], state["seed"],
                    state["pseudocount"], state["batch_size"],
                    patience=state.get('patience'), min_delta=state.get('min_delta', 1e-4))
        model.vocab_size = int(state["vocab_size"])
        model.pi_ = np.asarray(state["pi"], dtype=np.float64)
        model.A_ = np.asarray(state["A"], dtype=np.float64)
        model.B_ = np.asarray(state["B"], dtype=np.float64)
        model.history_ = state.get("history", [])
        model.best_epoch_ = state.get("best_epoch", model.n_iter)
        best_validation = state.get("best_validation_nll")
        model.best_validation_nll_ = (float("inf") if best_validation is None
                                      else float(best_validation))
        model.epochs_ran_ = state.get('epochs_ran', model.best_epoch_)
        model.stopped_early_ = state.get('stopped_early', False)
        model.maxbudget_reached_ = state.get('maxbudget_reached', False)
        return model


def model_from_state(state):
    """Restore a model from its plain dictionary state (no executable objects)."""
    classes = {"unigram": Unigram, "ngram": NGram, "hmm": CategoricalHMM}
    if state.get('type') == 'elman_rnn':
        from .rnn import ElmanLanguageModel
        classes['elman_rnn'] = ElmanLanguageModel
    if state.get("type") not in classes:
        raise ValueError("Unrecognized model type.")
    return classes[state["type"]].from_state(state)
