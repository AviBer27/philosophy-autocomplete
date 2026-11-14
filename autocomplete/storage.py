"""Portable JSON model artifacts; no executable pickle deserialization."""
import json
from pathlib import Path
import numpy as np
from .models import CategoricalHMM, NGram, Unigram


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def save_model(path, model, vocabulary, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format_version': 1, 'model_class': type(model).__name__,
               'vocabulary': vocabulary, 'metadata': metadata or {},
               'state': model.get_state()}
    path.write_text(json.dumps(payload, ensure_ascii=False, default=_json_default) + '\n', encoding='utf-8')


def load_model(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported model artifact format version.')
    registry = {c.__name__: c for c in (Unigram, NGram, CategoricalHMM)}
    name = payload.get('model_class')
    if name == 'ElmanLanguageModel':
        # Keep baseline/HMM loading usable when optional PyTorch is absent.
        from .rnn import ElmanLanguageModel
        registry[name] = ElmanLanguageModel
    if name not in registry:
        raise ValueError(f'Unknown model class: {name!r}')
    model = registry[name].from_state(payload['state'])
    vocabulary = payload['vocabulary']
    if len(vocabulary) != model.vocab_size or vocabulary[:2] != ['<unk>', '<eos>']:
        raise ValueError('Model vocabulary does not match the artifact vocabulary.')
    return model, vocabulary, payload.get('metadata', {})
