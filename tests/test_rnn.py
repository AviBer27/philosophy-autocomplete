"""Causality, masking and reproducibility checks for the optional Elman model."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
    from autocomplete.rnn import ElmanLanguageModel
except ImportError:
    torch = None

from autocomplete.storage import load_model, save_model


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is unavailable')
class ElmanTests(unittest.TestCase):
    sequences = [[2, 3, 1], [2, 4, 3, 1], [3, 1], [4, 1]]

    def model(self, **kwargs):
        config = {'hidden_size': 8, 'embedding_dim': 4, 'n_epochs': 0,
                  'seed': 9, 'batch_size': 3, 'verbose': False}
        config.update(kwargs)
        return ElmanLanguageModel(**config).fit(self.sequences, 5)

    def test_causal_full_sequence_predictions_match_incremental_prefixes(self):
        model = self.model()
        sequence = [2, 3, 4, 1]
        values = list(model.iter_distributions(sequence))
        for position, probabilities in enumerate(values):
            self.assertEqual(probabilities.shape, (5,))
            self.assertAlmostEqual(probabilities.sum(), 1.0)
            self.assertTrue(np.all(probabilities > 0))
            np.testing.assert_allclose(probabilities,
                                       model.next_distribution(sequence[:position]),
                                       rtol=1e-6, atol=1e-8)
        # Changing future targets must not change any preceding predictions.
        changed = list(model.iter_distributions([2, 3, 0, 0]))
        np.testing.assert_allclose(values[:3], changed[:3], rtol=1e-6, atol=1e-8)

    def test_first_word_is_predicted_from_distinct_bos_and_included_in_loss(self):
        model = self.model()
        self.assertNotEqual(model.bos_id_, model.pad_id_)
        self.assertGreaterEqual(model.bos_id_, model.vocab_size)
        expected = np.log(model.next_distribution([])[2])
        self.assertAlmostEqual(model.sequence_log_likelihood([2]), expected, places=7)
        self.assertEqual(model.sequence_log_likelihood([]), 0)
        # BOS is learned rather than serving as an output vocabulary word.
        before = model.next_distribution([])
        with torch.no_grad():
            model.network_.embedding.weight[model.bos_id_] += 2
        self.assertFalse(np.allclose(before, model.next_distribution([])))

    def test_padding_and_batch_size_do_not_change_valid_predictions_or_loss(self):
        model = self.model(batch_size=1)
        first = list(model.iter_sequence_distributions(self.sequences))
        loss = model.corpus_nll(self.sequences)
        model.batch_size = 4
        second = list(model.iter_sequence_distributions(self.sequences))
        self.assertAlmostEqual(loss, model.corpus_nll(self.sequences), places=7)
        for seq, left, right in zip(self.sequences, first, second):
            self.assertEqual(right.shape, (len(seq), 5))
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-8)
        outputs = list(model.iter_sequence_distributions([[], [2], []]))
        self.assertEqual([p.shape for p in outputs], [(0, 5), (1, 5), (0, 5)])

    def test_json_roundtrip_and_unknown_context(self):
        model = self.model(n_epochs=2)
        vocabulary = ['<unk>', '<eos>', 'a', 'b', 'c']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rnn.json'
            save_model(path, model, vocabulary, {'test': True})
            raw = json.loads(path.read_text())
            self.assertIsInstance(raw['state']['weights']['embedding.weight'], list)
            restored, words, metadata = load_model(path)
            self.assertEqual(words, vocabulary)
            self.assertEqual(metadata, {'test': True})
            np.testing.assert_allclose(model.next_distribution([2, 3]),
                                       restored.next_distribution([2, 3]), atol=0, rtol=0)
        np.testing.assert_allclose(model.next_distribution([999]),
                                   model.next_distribution([0]))

    def test_toy_training_reduces_loss_and_is_seed_reproducible(self):
        model = self.model(n_epochs=20, learning_rate=.03)
        self.assertLess(model.history_[-1]['train_nll'], model.history_[0]['train_nll'] * .65)
        repeated = self.model(n_epochs=20, learning_rate=.03)
        np.testing.assert_allclose(model.next_distribution([2]),
                                   repeated.next_distribution([2]), atol=0, rtol=0)

    def test_validation_stopping_restores_absolute_best_epoch(self):
        model = ElmanLanguageModel(hidden_size=8, embedding_dim=4, n_epochs=20,
            learning_rate=1e-12, patience=2, min_delta=10, verbose=False)
        model.fit(self.sequences, 5, self.sequences)
        self.assertEqual(model.epochs_ran_, 2)
        self.assertTrue(model.stopped_early_)
        self.assertFalse(model.maxbudget_reached_)
        losses = [row['validation_nll'] for row in model.history_]
        self.assertEqual(model.best_epoch_, int(np.argmin(losses)))
        self.assertAlmostEqual(model.corpus_nll(self.sequences), min(losses))


if __name__ == '__main__':
    unittest.main()
