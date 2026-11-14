"""Mathematical checks on tiny corpora; run with unittest discover."""

import itertools
import unittest

import numpy as np

from autocomplete.models import CategoricalHMM, NGram, Unigram, model_from_state


class HMMTests(unittest.TestCase):
    def known_model(self):
        model = CategoricalHMM(2, n_iter=0)
        model.vocab_size = 3
        model.pi_ = np.array([0.6, 0.4])
        model.A_ = np.array([[0.7, 0.3], [0.2, 0.8]])
        model.B_ = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
        model.history_ = []
        model.best_epoch_ = 0
        model.best_validation_nll_ = float("inf")
        return model

    def enumerate_probability(self, model, seq):
        total = 0.0
        for states in itertools.product(range(model.n_states), repeat=len(seq)):
            value = model.pi_[states[0]] * model.B_[states[0], seq[0]]
            for t in range(1, len(seq)):
                value *= model.A_[states[t - 1], states[t]] * model.B_[states[t], seq[t]]
            total += value
        return total

    def test_forward_and_prediction_match_path_enumeration(self):
        model = self.known_model()
        for seq in ([0], [2, 1], [0, 2, 1]):
            prefix_prob = self.enumerate_probability(model, seq)
            self.assertAlmostEqual(np.exp(model.sequence_log_likelihood(seq)), prefix_prob)
            expected = np.array([self.enumerate_probability(model, list(seq) + [w]) /
                                 prefix_prob for w in range(3)])
            np.testing.assert_allclose(model.next_distribution(seq), expected)
        np.testing.assert_allclose(model.next_distribution([]), model.pi_ @ model.B_)
        self.assertEqual(model.sequence_log_likelihood([]), 0)

    def test_unsmoothed_em_is_monotone_and_batches_match(self):
        corpus = [[0, 2, 2, 1], [2, 0, 1], [0, 1], [1], [2, 2, 1]]
        models = [CategoricalHMM(3, n_iter=7, seed=8, pseudocount=0,
                                 batch_size=b).fit(corpus, 3) for b in (1, 100)]
        for model in models:
            loss = np.array([row["train_nll"] for row in model.history_])
            self.assertTrue(np.all(np.diff(loss) < 1e-10), loss)
            np.testing.assert_allclose(model.A_.sum(axis=1), 1)
            np.testing.assert_allclose(model.B_.sum(axis=1), 1)
            self.assertAlmostEqual(model.pi_.sum(), 1)
        np.testing.assert_allclose(models[0].A_, models[1].A_)
        np.testing.assert_allclose(models[0].B_, models[1].B_)

    def test_expected_counts_match_exact_posterior_enumeration(self):
        model = self.known_model()
        corpus = [[0, 2, 1], [1]]
        expected_pi = np.zeros(2)
        expected_a = np.zeros((2, 2))
        expected_b = np.zeros((2, 3))
        for seq in corpus:
            evidence = self.enumerate_probability(model, seq)
            for states in itertools.product(range(2), repeat=len(seq)):
                weight = model.pi_[states[0]] * model.B_[states[0], seq[0]]
                for t in range(1, len(seq)):
                    weight *= model.A_[states[t - 1], states[t]] * model.B_[states[t], seq[t]]
                weight /= evidence
                expected_pi[states[0]] += weight
                for t, token in enumerate(seq):
                    expected_b[states[t], token] += weight
                    if t:
                        expected_a[states[t - 1], states[t]] += weight
        pi, a, b, log_likelihood = model._expectation([np.array(s) for s in corpus])
        np.testing.assert_allclose(pi, expected_pi)
        np.testing.assert_allclose(a, expected_a)
        np.testing.assert_allclose(b, expected_b)
        self.assertAlmostEqual(log_likelihood, sum(model.sequence_log_likelihood(s)
                                                   for s in corpus))

    def test_validation_restores_selected_epoch(self):
        corpus = [[0, 0, 1], [0, 1], [2, 1]]
        validation = [[2, 2, 1]]
        model = CategoricalHMM(2, 8, seed=4).fit(corpus, 3, validation)
        losses = [row["validation_nll"] for row in model.history_]
        self.assertEqual(model.best_epoch_, int(np.argmin(losses)))
        self.assertAlmostEqual(model.corpus_nll(validation), min(losses))

    def test_optional_early_stopping_preserves_best_checkpoint(self):
        corpus = [[0, 2, 1], [0, 1], [2, 2, 1]]
        model = CategoricalHMM(2, n_iter=20, patience=2, min_delta=10).fit(corpus, 3, corpus)
        self.assertTrue(model.stopped_early_)
        self.assertEqual(model.epochs_ran_, 2)
        self.assertFalse(model.maxbudget_reached_)
        losses = [row['validation_nll'] for row in model.history_]
        self.assertEqual(model.best_epoch_, int(np.argmin(losses)))
        self.assertAlmostEqual(model.corpus_nll(corpus), min(losses))
        restored = model_from_state(model.get_state())
        self.assertEqual(restored.patience, 2)
        self.assertEqual(restored.epochs_ran_, 2)
        self.assertTrue(restored.stopped_early_)

    def test_length_one_corpus_is_valid(self):
        model = CategoricalHMM(3, 3, pseudocount=0).fit([[0], [1], [2]], 3)
        np.testing.assert_allclose(model.A_.sum(axis=1), 1)
        np.testing.assert_allclose(model.B_.sum(axis=1), 1)


class SharedModelTests(unittest.TestCase):
    def test_normalization_serialization_and_online_context(self):
        corpus = [[2, 3, 1], [2, 2, 1], [3, 1]]
        models = [Unigram(), NGram(2), NGram(3), CategoricalHMM(2, 3)]
        for model in models:
            model.fit(corpus, 5)
            restored = model_from_state(model.get_state())
            seq = [2, 3, 1]
            for t, distribution in enumerate(model.iter_distributions(seq)):
                self.assertAlmostEqual(distribution.sum(), 1)
                self.assertTrue(np.all(distribution >= 0))
                np.testing.assert_allclose(distribution, model.next_distribution(seq[:t]))
                np.testing.assert_allclose(distribution, restored.next_distribution(seq[:t]))
            np.testing.assert_allclose(model.next_distribution([999]),
                                       model.next_distribution([0]))

    def test_ngram_respects_sentence_boundaries_and_unseen_backoff(self):
        model = NGram(2, interpolation=0.8).fit([[2, 1], [3, 1]], 5)
        # Both training sentence starts count equally, and EOS is not a start.
        starts = model.next_distribution([])
        self.assertGreater(starts[2], starts[1])
        self.assertAlmostEqual(starts[2], starts[3])
        np.testing.assert_allclose(model.next_distribution([4]),
                                   model.unigram_.probabilities_)

    def test_empty_training_data_is_rejected(self):
        for model in (Unigram(), NGram(), CategoricalHMM()):
            with self.assertRaises(ValueError):
                model.fit([[]], 3)


if __name__ == "__main__":
    unittest.main()
