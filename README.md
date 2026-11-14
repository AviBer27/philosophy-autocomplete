# Next-Word Prediction in Philosophy of Religion

A Bayesian Inference course project comparing **unigram, bigram, trigram, Hidden Markov Models trained with Baum-Welch, and Elman recurrent neural networks** on William James's *The Varieties of Religious Experience*.

[Read the five-page report](output/pdf/project_report.pdf) · [Measured results](output/results.csv) · [Open in Colab](https://colab.research.google.com/github/AviBer27/philosophy-autocomplete/blob/main/Philosophy_Autocomplete.ipynb)

The corpus and all eleven trained models are included. **Predictions work immediately after dependency installation; retraining is optional.** No API key or GPU is needed.

## Quick start

Use Python 3.12 (the tested version), or a compatible recent Python release. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py --text "religious experience is" --top-k 5
```

On Windows, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\activate`.

On the Mac where this project was prepared, the environment already exists:

```bash
cd "/Users/berezovskiyavi/Documents/ChatGPT/Bayesian"
./.venv/bin/python predict.py --text "religious experience is"
```

For an ongoing terminal session:

```bash
python predict.py --interactive
```

Type completed text and press Enter for suggestions. Type `:quit` to exit. This is a terminal interface; it updates after Enter rather than on each keystroke.

To compare saved models:

```bash
python predict.py --model output/models/trigram.json --text "the meaning of"
python predict.py --model output/models/hmm_16_s19.json --text "the meaning of"
python predict.py --model output/models/rnn_64_s7.json --text "the meaning of"
```

The default is the model selected by **validation negative log-likelihood**: `rnn_64_s7`. Optional `--prefix s` filters the next word to words beginning with `s`; `--json` prints machine-readable output. Input words, including the last one, are treated as complete. Sentence-ending punctuation resets context; abbreviations and decimal points are protected.

Displayed probabilities are original model probabilities over the full vocabulary. The few suggestions do not sum to 100%. Unknown words are mapped internally to `<unk>`; `<unk>` and `<eos>` are never offered as suggestions.

## Supplied experiment

The source contains 1,171 main-lecture paragraphs. Whole paragraphs are assigned in source order to training/validation/test (936/117/118), then split into independent sentence sequences. The vocabulary contains 2,000 symbols built from training text only. All models predict sentence starts and sentence endings with the same scoring rules.

| Model | Test perplexity | Top-3 accuracy | Top-5 accuracy |
|---|---:|---:|---:|
| Unigram | 204.85 | 13.59% | 18.60% |
| Bigram | 122.17 | 20.93% | 26.19% |
| Trigram | 127.46 | 22.32% | 27.65% |
| HMM: 16 states, seed 19 (validation-selected HMM) | 122.57 | 18.06% | 23.78% |
| RNN: 32 hidden units | 108.33 | 21.79% | 27.39% |
| RNN: 64 hidden units (validation-selected overall) | **106.58** | 22.07% | 27.65% |

The test contains 15,308 lexical targets and 15,890 encoded targets including EOS. The RNN wins on probability quality; the trigram has slightly higher Top-3 accuracy. The choice of metric therefore matters. These are results for this corpus and split, not a general ranking of model families.

Configurations were fixed before fitting:

- Unigram additive smoothing 0.1; bigram/trigram smoothing in {0.01, 0.1} and interpolation in {0.5, 0.8}, selected on validation NLL.
- HMM states {4, 8, 16}, seeds {7, 19}, at most 100 full-corpus Baum-Welch updates, smoothing 0.01, validation patience 10.
- Elman RNN hidden sizes {32, 64}, embedding dimension 32, seed 7, at most 25 epochs, Adam learning rate 0.003, batch size 64, validation patience 5, gradient norm clipping 5.
- All iterative models restore their absolute best validation checkpoint. The early-stopping minimum improvement is 0.0001 NLL.

All models use the preceding current sentence. The RNN is unidirectional and trained with full sentence backpropagation, a distinct BOS input, and masked right padding. It never sees future words for the prediction being scored. No overlapping windows are shuffled between subsets.

## What the scores mean

**Perplexity** is the exponential of mean negative log probability across all encoded tokens, including UNK and EOS. Lower is better. It is not a percentage and should not be compared directly between different vocabularies/tokenizers.

**Top-k accuracy** is the fraction of actual non-special target tokens among the k highest-ranked lexical candidates. Unknown targets remain in the denominator as misses; EOS targets are excluded. All lexical positions are evaluated without sampling. Every prediction is conditioned on the preceding true words, so this measures next-word prediction, not free-running generation.

The test OOV rate is 14.07%; those absent words cannot be suggested. Source-order splitting places later lectures in validation/test and creates topic shift. The historical prose, quotations, restricted vocabulary, heuristic sentence boundaries, limited model grids, and single RNN seed are limitations discussed in the report. An EM update and a neural epoch are different operations; their iteration counts do not represent equal compute.

## Reproduce or extend

```bash
python -m unittest discover -s tests -v
python experiment.py --output output_reproduced
```

The separate output directory preserves the supplied results. The default run writes an experiment specification before fitting, freezes all validation choices before test scoring, and saves models, complete metrics, curves, and examples. Runtime depends on the machine; the supplied run's retained fits total about 148 seconds on the recorded CPU environment, excluding final evaluation and baseline-search overhead.

A smaller exploratory run:

```bash
python experiment.py --states 4 --seeds 7 --iterations 5 --rnn-hidden 32 --rnn-epochs 2 --output output_small
```

To repeat on the optional Apollo corpus:

```bash
python experiment.py --corpus apollo --output output_apollo
```

The original Apollo-only experiment is retained in the author's workspace under `output/archive/apollo_v1`; that historical archive is omitted from the current submission package. Changing corpus, vocabulary, or training settings defines a new experiment. Any further tuning should use validation data rather than choosing settings from test scores.

## Files

| Path | Purpose |
|---|---|
| `autocomplete/data.py` | Corpus parsing, paragraph/transmission splitting, tokenization, training-only vocabulary. |
| `autocomplete/models.py` | NumPy n-grams and scaled forward-backward/Baum-Welch HMM. |
| `autocomplete/rnn.py` | PyTorch Elman network, full-sequence training, validation stopping. |
| `autocomplete/storage.py` | Portable JSON model weights; no executable pickle loading. |
| `experiment.py` | Prespecified model grid, validation selection, full held-out evaluation. |
| `predict.py` | One-shot and interactive terminal predictions. |
| `tests/` | Exact small HMM calculations, RNN causality/padding/persistence, split isolation, metric conventions. |
| `output/results.json` | Complete measured results, settings, histories and examples. |
| `output/validation_frozen.json` | Validation decisions captured before test scoring. |
| `output/parsed_paragraphs.json` | Source lines, lecture and subset for every retained paragraph. |
| `output/models/` | All eleven saved models, usable without retraining. |
| `scripts/build_report.py` | Reproducible academic PDF generation. |
| `Philosophy_Autocomplete.ipynb` | Optional Colab prediction and reproduction. |

The code is public at https://github.com/AviBer27/philosophy-autocomplete.

## Corpus provenance

William James, [*The Varieties of Religious Experience*, Project Gutenberg eBook 621](https://www.gutenberg.org/ebooks/621), downloaded 14 November 2026. The raw file retains the original Gutenberg notice. Main-lecture extraction excludes front matter, headings, endnotes, index and postscript; quotations within lectures remain. The exact source URL, SHA-256 digest and cleaning rules are in [data/SOURCE.md](data/SOURCE.md). The optional NASA source is documented there separately.

## Rebuild the report or ZIP

The finished PDF is included. Optional authoring dependencies are separate:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_report.py --repository-url https://github.com/AviBer27/philosophy-autocomplete
python scripts/package_project.py
```

To build a report from a fresh run:

```bash
python scripts/build_report.py --results output_reproduced/results.json --output output_reproduced/project_report.pdf --repository-url https://github.com/AviBer27/philosophy-autocomplete
```

The report is generated from the measured results for this implementation.
