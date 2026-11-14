"""Metric checks independent of the language-model implementations."""
import math
import unittest

import numpy as np

from experiment import evaluate


class FixedModel:
    def iter_distributions(self, sequence):
        # UNK has the largest probability, but cannot be a displayed word.
        for _ in sequence:
            yield np.array([0.40, 0.20, 0.15, 0.10, 0.08, 0.07])


class EvaluationTests(unittest.TestCase):
    def test_unknowns_are_misses_and_eos_only_affects_likelihood(self):
        result = evaluate(FixedModel(), [[2, 0, 3, 1]])
        self.assertEqual(result['lexical_tokens'], 3)
        self.assertEqual(result['tokens_including_eos'], 4)
        self.assertEqual(result['unknown_lexical_tokens'], 1)
        self.assertAlmostEqual(result['top1'], 1 / 3)
        self.assertAlmostEqual(result['top3'], 2 / 3)
        self.assertAlmostEqual(result['top5'], 2 / 3)
        expected = -sum(math.log(p) for p in [.15, .40, .10, .20]) / 4
        self.assertAlmostEqual(result['nll'], expected)
        self.assertAlmostEqual(result['perplexity'], math.exp(expected))

    def test_every_lexical_position_is_scored(self):
        result = evaluate(FixedModel(), [[2, 1], [5, 2, 1]])
        self.assertEqual(result['lexical_tokens'], 3)
        self.assertAlmostEqual(result['top1'], 2 / 3)
        self.assertAlmostEqual(result['top3'], 2 / 3)
        self.assertAlmostEqual(result['top5'], 1)


if __name__ == '__main__':
    unittest.main()
