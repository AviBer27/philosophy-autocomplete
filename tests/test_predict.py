"""Public prediction behavior, including safe saved-model roundtrips."""

import json
from pathlib import Path
import tempfile
import unittest

from autocomplete.models import CategoricalHMM, NGram, Unigram
from autocomplete.storage import load_model, save_model
from predict import suggest


class PredictionTests(unittest.TestCase):
    vocabulary = ['<unk>', '<eos>', 'roger', 'houston', 'ready', "we're"]
    sequences = [[2, 3, 1], [2, 4, 1], [3, 2, 1], [5, 4, 1]]

    def test_empty_input_offers_words_and_never_special_tokens(self):
        model = NGram(2).fit(self.sequences, len(self.vocabulary))
        result = suggest(model, self.vocabulary, '', top_k=20)
        self.assertEqual(result['context_words'], [])
        self.assertEqual(result['unknown_context_words'], [])
        words = [item['word'] for item in result['suggestions']]
        self.assertEqual(len(words), len(self.vocabulary) - 2)
        self.assertNotIn('<unk>', words)
        self.assertNotIn('<eos>', words)
        self.assertEqual(words[0], 'roger')

    def test_unknown_input_is_reported_and_mapped_to_unknown(self):
        model = CategoricalHMM(2, 2).fit(self.sequences, len(self.vocabulary))
        result = suggest(model, self.vocabulary, 'unseenword', top_k=20)
        self.assertEqual(result['unknown_context_words'], ['unseenword'])
        expected = model.next_distribution([0])
        for item in result['suggestions']:
            self.assertAlmostEqual(item['probability'],
                                   expected[self.vocabulary.index(item['word'])])

    def test_prefix_filters_without_renormalization_and_normalizes_quotes(self):
        model = Unigram().fit(self.sequences, len(self.vocabulary))
        result = suggest(model, self.vocabulary, 'ROGER', top_k=20, prefix='R')
        self.assertEqual(result['context_words'], ['roger'])
        self.assertEqual({item['word'] for item in result['suggestions']}, {'ready', 'roger'})
        expected = model.next_distribution([2])
        for item in result['suggestions']:
            self.assertAlmostEqual(item['probability'],
                                   expected[self.vocabulary.index(item['word'])])
        quoted = suggest(model, self.vocabulary, '', prefix='WE\u2019')
        self.assertEqual([item['word'] for item in quoted['suggestions']], ["we're"])
        self.assertEqual(suggest(model, self.vocabulary, '', prefix='zz')['suggestions'], [])

    def test_json_artifacts_preserve_public_predictions_for_all_model_types(self):
        models = [Unigram(), NGram(2), NGram(3), CategoricalHMM(2, 2)]
        with tempfile.TemporaryDirectory() as directory:
            for model in models:
                model.fit(self.sequences, len(self.vocabulary))
                path = Path(directory) / 'model.json'
                save_model(path, model, self.vocabulary, {'source': 'unit-test'})
                # The artifact contains plain JSON rather than pickled code.
                self.assertEqual(json.loads(path.read_text())['format_version'], 1)
                restored, vocabulary, metadata = load_model(path)
                self.assertEqual(vocabulary, self.vocabulary)
                self.assertEqual(metadata, {'source': 'unit-test'})
                before = suggest(model, vocabulary, 'roger', top_k=3)
                after = suggest(restored, vocabulary, 'roger', top_k=3)
                self.assertEqual(before, after)

    def test_finished_sentence_resets_context(self):
        model = NGram(3).fit(self.sequences, len(self.vocabulary))
        current = suggest(model, self.vocabulary, 'roger')
        after_boundary = suggest(model, self.vocabulary, 'Houston! Roger')
        self.assertEqual(after_boundary, current)
        self.assertEqual(suggest(model, self.vocabulary, 'Houston! '),
                         suggest(model, self.vocabulary, ''))

    def test_invalid_top_k_is_rejected(self):
        model = Unigram().fit(self.sequences, len(self.vocabulary))
        with self.assertRaises(ValueError):
            suggest(model, self.vocabulary, 'roger', top_k=0)


if __name__ == '__main__':
    unittest.main()
