"""Train, select on validation data, then evaluate the frozen experiment on test.

Run: python experiment.py
The bundled corpus makes this command fully offline after dependency installation.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
from autocomplete.data import context_tokens, prepare_corpus
from autocomplete.models import CategoricalHMM, NGram, Unigram
from autocomplete.storage import save_model


def evaluate(model, sequences):
    """Teacher-forced metrics, with every lexical OOV target counted as a miss."""
    log_probability = 0.0
    tokens = lexical = unknown = 0
    hits = {1: 0, 3: 0, 5: 0}
    distributions = (model.iter_sequence_distributions(sequences)
                     if hasattr(model, 'iter_sequence_distributions')
                     else (model.iter_distributions(sequence) if hasattr(model, 'iter_distributions')
                           else (model.next_distribution(sequence[:i]) for i in range(len(sequence)))
                           for sequence in sequences))
    for sequence, iterator in zip(sequences, distributions):
        for target, probabilities in zip(sequence, iterator):
            probability = float(probabilities[target])
            if probability <= 0 or not math.isfinite(probability):
                raise ValueError('A model assigned invalid probability to an observed target.')
            log_probability += math.log(probability)
            tokens += 1
            if target == 1:  # EOS is a language-model target, not a displayed word.
                continue
            lexical += 1
            unknown += target == 0
            # IDs break probability ties deterministically. UNK/EOS are never suggestions.
            rank = np.lexsort((np.arange(2, len(probabilities)), -probabilities[2:]))[:5] + 2
            for k in hits:
                hits[k] += int(target != 0 and target in rank[:k])
    if tokens != sum(map(len, sequences)) or not tokens or not lexical:
        raise ValueError('Evaluation requires one predictive distribution per token and lexical targets.')
    nll = -log_probability / tokens
    return {'nll': nll, 'perplexity': math.exp(nll),
            **{f'top{k}': hits[k] / lexical for k in hits},
            'tokens_including_eos': tokens, 'lexical_tokens': lexical,
            'unknown_lexical_tokens': unknown}


def nll_only(model, sequences):
    if hasattr(model, 'corpus_nll'):
        return model.corpus_nll(sequences)
    return -sum(model.sequence_log_likelihood(s) for s in sequences) / sum(map(len, sequences))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def parameter_count(model):
    """Probability scalars / neural weights, excluding context-index storage."""
    if hasattr(model, 'n_parameters_'):
        return model.n_parameters_
    if isinstance(model, CategoricalHMM):
        k, v = model.n_states, model.vocab_size
        return k + k*k + k*v
    if isinstance(model, NGram):
        return model.vocab_size + sum(len(ids) for contexts in model.rows_.values()
                                     for ids, _ in contexts.values())
    return model.vocab_size


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    corpus = prepare_corpus(args.raw, args.max_vocab, corpus=args.corpus)
    vocabulary = corpus['vocabulary']
    train, validation, test = (corpus[k] for k in ('train', 'validation', 'test'))
    models, rows, search = {}, [], []
    print(json.dumps(corpus['metadata'], indent=2), flush=True)
    specification = {'corpus': args.corpus, 'source_sha256': corpus['metadata']['source']['sha256'],
        'max_vocab': args.max_vocab, 'hmm_states': args.states, 'hmm_seeds': args.seeds,
        'hmm_iterations': args.iterations, 'hmm_patience': args.hmm_patience,
        'rnn_hidden': args.rnn_hidden, 'rnn_seeds': args.rnn_seeds, 'rnn_epochs': args.rnn_epochs,
        'rnn_patience': args.rnn_patience, 'rnn_embedding': args.rnn_embedding,
        'rnn_learning_rate': .003, 'rnn_batch_size': 64, 'min_delta': .0001,
        'baseline_grid': {'alpha': [.01, .1], 'interpolation': [.5, .8]},
        'protocol': 'Predeclared before fitting; all choices and checkpoints frozen by validation before the final held-out evaluation.'}
    write_json(output / 'experiment_specification.json', specification)

    def retain(name, family, model, parameters, seconds):
        validation_metrics = evaluate(model, validation)
        row = {'name': name, 'family': family, 'parameters': parameters,
               'training_seconds': seconds, 'validation': validation_metrics,
               'history': getattr(model, 'history_', []),
               'best_epoch': getattr(model, 'best_epoch_', None),
               'epochs_ran': getattr(model, 'epochs_ran_', None),
               'stopped_early': getattr(model, 'stopped_early_', False),
               'maxbudget_reached': getattr(model, 'maxbudget_reached_', False),
               'parameter_count': parameter_count(model)}
        models[name] = model
        rows.append(row)
        print(f'{name}: validation PPL={validation_metrics["perplexity"]:.3f}; train={seconds:.2f}s', flush=True)

    start = time.perf_counter()
    unigram = Unigram(alpha=.1).fit(train, len(vocabulary))
    retain('unigram', 'unigram', unigram, {'alpha': .1}, time.perf_counter()-start)
    for order, name in ((2, 'bigram'), (3, 'trigram')):
        candidates = []
        search_start = time.perf_counter()
        for alpha in (.01, .1):
            for weight in (.5, .8):
                start = time.perf_counter()
                model = NGram(order=order, alpha=alpha, interpolation=weight).fit(train, len(vocabulary))
                seconds = time.perf_counter() - start
                val_nll = nll_only(model, validation)
                parameters = {'order': order, 'alpha': alpha, 'interpolation': weight}
                search.append({'family': name, 'parameters': parameters, 'validation_nll': val_nll})
                candidates.append((val_nll, model, parameters, seconds))
        _, winner, parameters, seconds = min(candidates, key=lambda item: item[0])
        retain(name, name, winner, parameters, seconds)
        rows[-1]['search_seconds'] = time.perf_counter() - search_start

    for states in args.states:
        for seed in args.seeds:
            name = f'hmm_{states}_s{seed}'
            print(f'Training {name} (budget {args.iterations} EM updates, patience {args.hmm_patience})...', flush=True)
            start = time.perf_counter()
            model = CategoricalHMM(n_states=states, n_iter=args.iterations, seed=seed,
                pseudocount=.01, batch_size=128, patience=args.hmm_patience,
                min_delta=.0001, verbose=True).fit(train, len(vocabulary), validation_sequences=validation)
            seconds = time.perf_counter() - start
            retain(name, 'HMM', model, {'n_states': states, 'seed': seed, 'n_iter': args.iterations,
                'pseudocount': .01, 'batch_size': 128, 'patience': args.hmm_patience, 'min_delta': .0001,
                'free_parameters': (states-1)+states*(states-1)+states*(len(vocabulary)-1)}, seconds)

    if args.rnn_hidden:
        from autocomplete.rnn import ElmanLanguageModel
        for hidden in args.rnn_hidden:
            for seed in args.rnn_seeds:
                name = f'rnn_{hidden}_s{seed}'
                parameters = {'hidden_size': hidden, 'embedding_dim': args.rnn_embedding,
                    'seed': seed, 'n_epochs': args.rnn_epochs, 'batch_size': 64,
                    'learning_rate': .003, 'patience': args.rnn_patience, 'min_delta': .0001,
                    'architecture': 'Unidirectional Elman tanh RNN, embedding and linear-softmax output; full-sentence BPTT; gradient clipping norm 5.'}
                print(f'Training {name} (budget {args.rnn_epochs} epochs, patience {args.rnn_patience})...', flush=True)
                start = time.perf_counter()
                model = ElmanLanguageModel(hidden_size=hidden, embedding_dim=args.rnn_embedding,
                    n_epochs=args.rnn_epochs, seed=seed, batch_size=64, learning_rate=.003,
                    patience=args.rnn_patience, min_delta=.0001, verbose=True).fit(
                        train, len(vocabulary), validation_sequences=validation)
                retain(name, 'RNN', model, parameters, time.perf_counter()-start)

    # Freeze every hyperparameter and checkpoint BEFORE any test-set evaluation.
    selected = min(rows, key=lambda row: row['validation']['nll'])['name']
    selected_hmm = min((r for r in rows if r['family'] == 'HMM'),
                       key=lambda row: row['validation']['nll'])['name']
    selected_count = min((r for r in rows if r['family'] in ('unigram', 'bigram', 'trigram')),
                         key=lambda row: row['validation']['nll'])['name']
    rnn_rows = [r for r in rows if r['family'] == 'RNN']
    selected_rnn = min(rnn_rows, key=lambda row: row['validation']['nll'])['name'] if rnn_rows else None
    selection = {'name': selected, 'path': str(output / 'models' / f'{selected}.json'),
                 'criterion': 'Minimum validation NLL over predeclared model variants.'}
    write_json(output / 'selected_model.json', selection)
    write_json(output / 'validation_frozen.json', {'selected_model': selection,
        'selected_hmm': selected_hmm, 'selected_rnn': selected_rnn, 'selected_count': selected_count,
        'models': [{key: value for key, value in row.items() if key != 'test'} for row in rows],
        'test_evaluation_started': False})
    for row in rows:
        name = row['name']
        row['test'] = evaluate(models[name], test)
        model_path = output / 'models' / f'{name}.json'
        save_model(model_path, models[name], vocabulary,
                   {'name': name, 'corpus': corpus['metadata'], 'parameters': row['parameters']})
        row['artifact_path'] = str(model_path)
        row['artifact_bytes'] = model_path.stat().st_size
        print(f'{name}: TEST PPL={row["test"]["perplexity"]:.3f}, top3={100*row["test"]["top3"]:.2f}%', flush=True)

    examples = []
    contexts = (('religious experience is', 'the meaning of', 'the human mind', 'we must remember that', 'a sense of')
                if args.corpus == 'james' else ('houston we have', 'we are go for', 'reading you loud and', 'stand by for', 'the lunar module'))
    for context in contexts:
        ids = [{w: i for i, w in enumerate(vocabulary)}.get(w, 0) for w in context_tokens(context)]
        predictions = {}
        for name in dict.fromkeys(n for n in (selected, selected_count, selected_hmm, selected_rnn) if n is not None):
            probabilities = models[name].next_distribution(ids)
            order = np.lexsort((np.arange(2, len(vocabulary)), -probabilities[2:]))[:5] + 2
            predictions[name] = [{'word': vocabulary[i], 'probability': float(probabilities[i])} for i in order]
        examples.append({'context': context, 'predictions': predictions})
    result = {'config': {**specification, 'raw_path': str(args.raw or ('data/raw/james_varieties.txt' if args.corpus == 'james' else 'data/raw/apollo11_technical.html')),
               'python': sys.version.split()[0], 'numpy': np.__version__, 'platform': platform.platform()},
        'corpus': corpus['metadata'], 'metric_definitions': {
            'nll': 'Mean negative natural log conditional probability of all tokens, including UNK and EOS.',
            'perplexity': 'exp(nll), evaluated on the same fixed train-only vocabulary for every model.',
            'top1_top3_top5': 'Fraction of lexical targets among the top k lexical suggestions; UNK targets count as misses; EOS targets excluded; no sampled positions.',
            'teacher_forcing': 'Every prediction sees only preceding true words within its sentence; state resets for every sentence.',
            'selection': 'Validation only; HMM/RNN absolute best epoch restored on validation NLL. Early-stopping patience uses minimum NLL improvement 0.0001. Test run only after all choices frozen.',
            'context': 'Same complete preceding sentence available to all models; no cross-sentence state or fixed context-window ablation. N-grams inherently use only order-1 preceding tokens.',
            'parameter_count': 'Number of stored model probability scalars for count/HMM models, or learned neural weight scalars for RNN; context-ID storage excluded. HMM constrained free-parameter count is also given.',
            'training_seconds': 'Measured wall time of the retained fit including per-epoch train/validation scoring; baseline search cost recorded separately. History elapsed_seconds is cumulative within that fit.',
            'test_comparisons': 'All prespecified HMM/RNN variants and validation-selected baseline per order are shown descriptively; test scores do not select a model.'},
        'validation_search': search, 'models': rows, 'selected_model': selection,
        'selected_hmm': selected_hmm, 'selected_rnn': selected_rnn, 'selected_count': selected_count, 'examples': examples}
    if rnn_rows:
        import torch
        result['config']['torch'] = torch.__version__
    write_json(output / 'results.json', result)
    write_json(output / 'corpus_metadata.json', corpus['metadata'])
    write_json(output / ('parsed_turns.json' if args.corpus == 'apollo' else 'parsed_paragraphs.json'), corpus['groups'])
    with (output / 'results.csv').open('w', newline='') as handle:
        fields = ['model', 'validation_perplexity', 'test_perplexity', 'test_top1', 'test_top3', 'test_top5', 'training_seconds', 'parameter_count', 'best_epoch', 'epochs_ran', 'stopped_early', 'maxbudget_reached']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({'model': row['name'], 'validation_perplexity': row['validation']['perplexity'],
                'test_perplexity': row['test']['perplexity'], 'test_top1': row['test']['top1'],
                'test_top3': row['test']['top3'], 'test_top5': row['test']['top5'],
                **{key: row[key] for key in ('training_seconds', 'parameter_count', 'best_epoch', 'epochs_ran', 'stopped_early', 'maxbudget_reached')}})
    print(f'Selected {selected} by validation NLL. Results: {output / "results.json"}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', choices=['james', 'apollo'], default='james')
    parser.add_argument('--raw', help='Override the selected corpus raw file.')
    parser.add_argument('--output', default='output')
    parser.add_argument('--max-vocab', type=int, default=2000)
    parser.add_argument('--states', nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 19])
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--hmm-patience', type=int, default=10)
    parser.add_argument('--rnn-hidden', nargs='*', type=int, default=[32, 64], help='Hidden sizes; pass flag alone to skip optional RNN runs.')
    parser.add_argument('--rnn-seeds', nargs='+', type=int, default=[7])
    parser.add_argument('--rnn-epochs', type=int, default=25)
    parser.add_argument('--rnn-embedding', type=int, default=32)
    parser.add_argument('--rnn-patience', type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.hmm_patience, args.rnn_epochs, args.rnn_embedding, args.rnn_patience) < 1 or any(k < 1 for k in args.states+args.rnn_hidden):
        parser.error('Budgets, patience, states and hidden sizes must be positive.')
    run(args)


if __name__ == '__main__':
    main()
