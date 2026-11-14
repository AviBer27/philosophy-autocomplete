"""Corpus cleaning and train-only vocabulary construction.

A sentence is an independent HMM sequence. The final EOS token is observed;
no transitions join sentences or cross a train/validation/test boundary.
"""
from collections import Counter
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re
import unicodedata

UNK = '<unk>'
EOS = '<eos>'
WORD_RE = re.compile(r"\d+(?:\.\d+)+|[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)
ABBREVIATION_RE = re.compile(r'\b(Mr|Mrs|Ms|Dr|St|Sr|Jr|Capt|Col|Gen|Rev|Prof|Cf|Viz|Vol|Vs|Pp)\.', re.I)
OPENING = 'It is a truth universally acknowledged'
CORPUS_PATHS = {'james': 'data/raw/james_varieties.txt', 'apollo': 'data/raw/apollo11_technical.html'}


def normalize_text(text):
    return (unicodedata.normalize('NFC', text).replace('\u2019', "'").replace('\u2018', "'")
            .replace('\u2010', '-').replace('\u2011', '-'))


def tokenize(text):
    """Lowercase words, numbers and codes; retain apostrophes/hyphens/decimals."""
    return WORD_RE.findall(normalize_text(text).lower())


def strip_bracketed(text):
    """Remove bracketed illustration captions, including nested brackets."""
    depth, result = 0, []
    for char in text:
        if char == '[':
            depth += 1
        elif char == ']' and depth:
            depth -= 1
        elif not depth:
            result.append(char)
    return ''.join(result)


def clean_gutenberg(text):
    """Extract the actual novel (not publisher material, preface, or PG license)."""
    if OPENING in text:
        text = text[text.index(OPENING):]
    elif '*** START OF' in text:
        text = re.split(r'\*\*\* START OF[^\n]*\n', text, maxsplit=1)[-1]
    text = re.split(r'\*\*\* END OF', text, maxsplit=1)[0]
    text = strip_bracketed(text)
    text = re.sub(r'^\s*CHAPTER\s+[IVXLCDM\d]+\.?\s*$', '', text,
                  flags=re.MULTILINE | re.IGNORECASE)
    return normalize_text(text)


def _protect_periods(text):
    protected = ABBREVIATION_RE.sub(lambda m: m.group(0)[:-1] + '\ue000', normalize_text(text))
    protected = re.sub(r'(?<=\d)\.(?=\d)', '\ue001', protected)
    # Names and acronyms are frequent in academic prose. This remains a heuristic.
    protected = re.sub(r'\b(?:[A-Z]\.[ \t]*){2,}', lambda m: m.group(0).replace('.', '\ue000'), protected)
    protected = re.sub(r'\b[A-Z]\.(?=\s+[A-Z][a-z])', lambda m: m.group(0).replace('.', '\ue000'), protected)
    protected = re.sub(r'\b(?:e\.g|i\.e)\.', lambda m: m.group(0).replace('.', '\ue000'), protected, flags=re.I)
    return protected


def split_sentences(text):
    """Split .!? after protecting decimals, common abbreviations and initials."""
    protected = _protect_periods(text)
    return [words for part in re.split(r'[.!?]+', protected)
            if (words := tokenize(part.replace('\ue000', '.').replace('\ue001', '.')))]


def context_tokens(text):
    """Words in the current sentence; trailing words count as complete input."""
    protected = _protect_periods(text)
    return tokenize(re.split(r'[.!?]+', protected)[-1].replace('\ue000', '.').replace('\ue001', '.'))


def parse_james_paragraphs(text):
    """Extract only the twenty main lectures as whole source paragraphs.

    Contents/preface precede the exact uppercase Lecture I heading; postscript,
    index and endnotes follow the uppercase POSTSCRIPT heading. Attached numeric
    footnote references are stripped, but spaced numbered list labels are kept.
    Every paragraph retains its raw source line range and lecture heading.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == 'LECTURE I. RELIGION AND NEUROLOGY.'), None)
    if start is None:
        raise ValueError('James corpus missing its exact first lecture heading.')
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == 'POSTSCRIPT.'), None)
    if end is None:
        raise ValueError('James corpus missing its POSTSCRIPT end delimiter.')
    paragraphs, current, lecture, first_line = [], [], None, None

    def flush(last_line):
        nonlocal current, first_line
        if current:
            paragraph = normalize_text('\n'.join(current))
            paragraph = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '-', paragraph)
            paragraph = re.sub(r'(?<=\S)\(\d+\)', '', paragraph)
            paragraph = ' '.join(paragraph.split())
            if tokenize(paragraph):
                paragraphs.append({'paragraph_index': len(paragraphs), 'lecture': lecture,
                    'source_line_start': first_line, 'source_line_end': last_line, 'text': paragraph})
        current, first_line = [], None

    for index in range(start, end):
        line = lines[index]
        if re.match(r'^LECTURES?\s+[IVX]+[., ]', line):
            flush(index)
            lecture = normalize_text(line.strip())
        elif not line.strip() or not re.search(r'\w', line):
            flush(index)
        else:
            if first_line is None:
                first_line = index + 1
            current.append(line)
    flush(end)
    return paragraphs


class _TranscriptHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('head', 'script', 'style'):
            self.skip = True
        if tag in ('br', 'p', 'tr', 'hr'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('head', 'script', 'style'):
            self.skip = False
        if tag in ('p', 'tr', 'td'):
            self.parts.append('\n')

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def parse_apollo_turns(html):
    """Extract timestamped raw NASA transmissions, retaining source metadata.

    OCR variants in timestamp fields (O/I/comma/??) are accepted as delimiters,
    preserved verbatim in metadata, and never fed to a language model. The text
    itself is not hand-corrected. Source order defines the chronological split.
    """
    parser = _TranscriptHTMLParser()
    parser.feed(html)
    timestamp = re.compile(r'^([0-9OoIlhHkK]{2}\s+[0-9OoIlhHkK]{2}\s+[0-9OoIlhHkK]{2}\s+[0-9OoIl,?]{1,4})(?:\s+(.*))?$')
    turns, current = [], None
    for line in ''.join(parser.parts).splitlines():
        line = line.strip()
        if not line:
            continue
        match = timestamp.match(line)
        if match:
            if current is not None:
                current['text'] = ' '.join(current.pop('parts'))
                if tokenize(current['text']):
                    turns.append(current)
            current = {'timestamp': match.group(1), 'speaker': (match.group(2) or 'UNKNOWN').strip(), 'parts': []}
            continue
        if re.match(r'^\((EAGLE|TRANQ|COLUMBIA|EVA)\s*\)[A-Z]?$', line):
            if current is not None and not current['parts']:
                current['speaker'] += ' ' + line
            continue
        if (re.match(r'^\(GO\w+ NET.*\)\s*Tape', line, re.I)
            or re.search(r'\(REV\s*\d', line, re.I) or 'BEGIN LUNAR REV' in line
            or line == 'END OF TAPE' or line.startswith('***')
            or line == 'NOTE' or line.startswith('Subsequent to TLI, there is continuous acquisition')
            or line.startswith('NO SCRIPT (PDF-File corrupt)')
            or line.startswith('REST PERIOD')
            or re.match(r'^APOLLO\s+11\s+AIR-TO-.*VOICE TRANSCRIPTION', line, re.I)):
            continue
        if current is not None:
            current['parts'].append(line)
    if current is not None:
        current['text'] = ' '.join(current.pop('parts'))
        if tokenize(current['text']):
            turns.append(current)
    for i, turn in enumerate(turns):
        turn['turn_index'] = i
    return turns


def build_vocabulary(sentences, max_vocab=2000):
    if max_vocab < 3:
        raise ValueError('max_vocab must be at least 3 (UNK, EOS, and one word).')
    counts = Counter(word for sentence in sentences for word in sentence)
    # Stable lexical tie breaking makes regeneration independent of hash seeds.
    return [UNK, EOS] + [word for word, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_vocab-2]]


def encode(sentences, vocabulary):
    lookup = {word: i for i, word in enumerate(vocabulary)}
    return [[lookup.get(word, 0) for word in sentence] + [1] for sentence in sentences]


def prepare_corpus(raw_path=None, max_vocab=2000, corpus=None):
    """Load James by default; preserve Apollo and custom plain-text options."""
    if corpus is None:
        corpus = ('james' if raw_path is None or Path(raw_path).name == 'james_varieties.txt'
                  else 'apollo' if Path(raw_path).suffix.lower() == '.html' else 'custom')
    if corpus not in ('james', 'apollo', 'custom'):
        raise ValueError('corpus must be james, apollo or custom')
    raw_path = Path(raw_path or CORPUS_PATHS[corpus])
    raw = raw_path.read_bytes()
    source_common = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    if corpus == 'james':
        groups = parse_james_paragraphs(raw.decode('utf-8-sig'))
        grouping, plural = 'paragraph', 'paragraphs'
        source = {'title': 'The Varieties of Religious Experience: A Study in Human Nature',
            'author': 'William James', 'catalog_url': 'https://www.gutenberg.org/ebooks/621',
            'download_url': 'https://www.gutenberg.org/cache/epub/621/pg621.txt',
            'download_date': '2026-11-14', 'license': 'Project Gutenberg; public domain in the USA. Full source notice retained in the raw text.', **source_common}
        cleaning = ('Keep the twenty main lectures from the exact uppercase Lecture I heading to before POSTSCRIPT. '
            'Exclude preface, contents, postscript, index, endnotes and Gutenberg wrappers. Remove lecture headings, '
            'decorative separators and attached numeric footnote references. Keep quotations and numbered lists in the lectures. '
            'Join wrapped lines inside whole source paragraphs; normalize typographic hyphens.')
    elif corpus == 'apollo':
        groups = parse_apollo_turns(raw.decode('cp1252'))
        grouping, plural = 'turn', 'turns'
        source = {'title': 'Apollo 11 Technical Air-to-Ground Voice Transcription',
            'author': 'NASA; raw OCR HTML courtesy Thomas Schwagmeier',
            'catalog_url': 'https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11trans.html',
            'download_url': 'https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11transcript_tec.html',
            'download_date': '2026-11-12', 'license': 'NASA raw transcripts: no copyright asserted by official catalog; corrected Lunar Surface Journal excluded.', **source_common}
        cleaning = ('Extract timestamped transmissions; exclude introduction, time/speaker fields, tape/page/station/revolution '
            'headings, coverage/rest-period annotations and clipping footnotes. Preserve source OCR wording without hand correction.')
    else:
        text = clean_gutenberg(raw.decode('utf-8-sig'))
        blocks = re.split(r'\n\s*\n', text) if re.search(r'\n\s*\n', text) else [' '.join(s) for s in split_sentences(text)]
        groups = [{'paragraph_index': i, 'text': ' '.join(block.split())} for i, block in enumerate(blocks) if tokenize(block)]
        grouping, plural = 'paragraph', 'paragraphs'
        source = {'title': 'Custom plain-text corpus', **source_common}
        cleaning = 'Generic plain text; optional Project Gutenberg wrappers removed.'
    n = len(groups)
    first, second = int(.8*n), int(.9*n)
    if first < 1 or second <= first or n <= second:
        raise ValueError('Corpus must contain enough source groups for nonempty 80/10/10 splits.')
    group_splits = {'train': groups[:first], 'validation': groups[first:second], 'test': groups[second:]}
    splits = {name: [sentence for group in items for sentence in split_sentences(group['text'])]
              for name, items in group_splits.items()}
    vocabulary = build_vocabulary(splits['train'], max_vocab)
    encoded = {name: encode(items, vocabulary) for name, items in splits.items()}
    metadata = {'corpus_id': corpus, 'source': source, 'grouping': grouping,
        'split': f'Source-order whole {plural}; floor(0.8*N), floor(0.9*N). Split each assigned {grouping} into sentence sequences afterward; no source {grouping} spans subsets.',
        'vocabulary_size': len(vocabulary), 'requested_max_vocab': max_vocab,
        'tokenization': ('NFC lowercase words and numbers; retain internal apostrophes/hyphens, alphanumeric codes and decimal numbers. '
            'Protect common abbreviations, initials/acronyms and decimals before heuristic .!? sentence splitting. Append EOS per sentence.'),
        'cleaning': cleaning, 'total_groups': n, 'total_' + plural: n}
    for name, sequences in encoded.items():
        if not sequences:
            raise ValueError(f'No tokenized sentences in {name} split')
        lexical = sum(len(s)-1 for s in sequences)
        unknown = sum(s.count(0) for s in sequences)
        items = group_splits[name]
        first_group, last_group = items[0], items[-1]
        metadata[name] = {'sentences': len(sequences), 'lexical_tokens': lexical,
            'tokens_including_eos': sum(map(len, sequences)), 'unknown_tokens': unknown,
            'oov_rate': unknown / lexical, 'max_sequence_length_including_eos': max(map(len, sequences)),
            'groups': len(items), plural: len(items),
            'first_' + grouping + '_index': first_group.get(grouping + '_index', 0),
            'last_' + grouping + '_index': last_group.get(grouping + '_index', len(items)-1)}
        if corpus == 'apollo':
            metadata[name].update(first_timestamp=first_group['timestamp'], last_timestamp=last_group['timestamp'])
        if corpus == 'james':
            metadata[name].update(first_lecture=first_group['lecture'], last_lecture=last_group['lecture'],
                first_source_line=first_group['source_line_start'], last_source_line=last_group['source_line_end'])
        for group in items:
            group['split'] = name
    return {'vocabulary': vocabulary, 'metadata': metadata, 'groups': groups, plural: groups, **encoded}
