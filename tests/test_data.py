import tempfile
import unittest
from pathlib import Path
from autocomplete.data import build_vocabulary, clean_gutenberg, context_tokens, encode, parse_apollo_turns, parse_james_paragraphs, prepare_corpus, split_sentences, tokenize

class DataTests(unittest.TestCase):
    def test_vocabulary_uses_only_supplied_training_words(self):
        vocabulary = build_vocabulary([['cat', 'sat'], ['cat']], 4)
        self.assertEqual(vocabulary, ['<unk>', '<eos>', 'cat', 'sat'])
        self.assertEqual(encode([['unseen']], vocabulary), [[0, 1]])

    def test_boundaries_abbreviations_and_unicode(self):
        self.assertEqual(split_sentences("Mr. Bennet can't go. Why? He’s here!"),
                         [['mr', 'bennet', "can't", 'go'], ['why'], ["he's", 'here']])
        self.assertEqual(context_tokens('He went. Mr. Bennet'), ['mr', 'bennet'])
        self.assertEqual(context_tokens('He went! '), [])

    def test_strip_frontmatter_and_nested_illustration(self):
        text = 'preface\nIt is a truth universally acknowledged. [Illustration: [nested] caption]\nCHAPTER II.\nStory.\n*** END OF THE PROJECT GUTENBERG EBOOK\nlicense'
        cleaned = clean_gutenberg(text)
        self.assertNotIn('preface', cleaned)
        self.assertNotIn('caption', cleaned)
        self.assertNotIn('CHAPTER', cleaned)
        self.assertNotIn('license', cleaned)

    def test_chronological_split_and_no_vocabulary_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tiny.txt'
            path.write_text('one. '*8 + 'validationword. testword.')
            data = prepare_corpus(path, 20)
            self.assertEqual([len(data[k]) for k in ('train', 'validation', 'test')], [8, 1, 1])
            self.assertNotIn('validationword', data['vocabulary'])
            self.assertNotIn('testword', data['vocabulary'])
            self.assertEqual(data['test'], [[0, 1]])

    def test_numeric_codes_and_decimal_sentence_boundaries(self):
        self.assertEqual(tokenize("Apollo 11's P52 S-IVB 101.4"),
                         ['apollo', "11's", 'p52', 's-ivb', '101.4'])
        self.assertEqual(split_sentences('Altitude 101.4. Go!'), [['altitude', '101.4'], ['go']])
        self.assertEqual(context_tokens('Go. Altitude 101.4'), ['altitude', '101.4'])

    def test_apollo_metadata_never_enters_transmission_text(self):
        html = '''INTRODUCTION<br>Not spoken.
        <br>00 00 00 04 <font>CDR</font><br>Roger. Clock.<br><br>
        (GOSS NET 1) Tape 1/2 Page 2<br>MILA (REV 1)<br>
        VANGUARD (REV2)<br>APOLLO 11 AIR-TO-GHOUND VOICE TRANSCRIPTION<br>
        (GODS NET 1) Tape 84/8 Page 497<br>REST PERIOD - NO C0MMUNICATIONS<br>
        NO SCRIPT (PDF-File corrupt)<br>
        NOTE<br>Subsequent to TLI, there is continuous acquisition among Goldstone (GDS).<br>
        00 00 00 13 <font>CC</font><br>Go for staging.<br>
        03 04 02 -- BEGIN LUNAR REV 1<br>END OF TAPE<br>
        00 00 19 O1 <font>CDR</font><br>Copy 101.4.<br>
        *** Three asterisks denote clipping of words and phrases.'''
        turns = parse_apollo_turns(html)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0]['text'], 'Roger. Clock.')
        self.assertEqual(turns[1]['text'], 'Go for staging.')
        self.assertEqual(turns[2]['timestamp'], '00 00 19 O1')
        self.assertEqual(turns[2]['text'], 'Copy 101.4.')

    def test_turn_split_precedes_sentence_expansion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tiny.html'
            path.write_text(''.join(f'<br>00 00 00 {i:02d} <font>CC</font><br>Word{i}. More{i}.' for i in range(10)))
            data = prepare_corpus(path, 100)
            self.assertEqual([len(data[k]) for k in ('train', 'validation', 'test')], [16, 2, 2])
            self.assertEqual(data['metadata']['validation']['first_turn_index'], 8)
            self.assertNotIn('word8', data['vocabulary'])
            self.assertEqual([t['split'] for t in data['turns']], ['train']*8+['validation', 'test'])

    def test_james_excludes_front_and_back_matter_and_preserves_groups(self):
        text = ('CONTENTS\nLecture I. Religion And Neurology.\nPreface material.\n\n'
                'LECTURE I. RELIGION AND NEUROLOGY.\n'
                'Main lecture text.(1)\nWrapped line continues.\n\n'
                '------------------\n\n'
                'LECTURE XX. CONCLUSIONS.\nSecond paragraph: (1) list item.\n\n'
                'POSTSCRIPT.\nExcluded postscript.\nINDEX.\nExcluded index.\nFOOTNOTES\n1 Endnote.')
        paragraphs = parse_james_paragraphs(text)
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0]['text'], 'Main lecture text. Wrapped line continues.')
        self.assertEqual(paragraphs[1]['text'], 'Second paragraph: (1) list item.')
        self.assertEqual(paragraphs[0]['paragraph_index'], 0)
        self.assertEqual(paragraphs[0]['source_line_start'], 6)
        self.assertEqual(paragraphs[1]['lecture'], 'LECTURE XX. CONCLUSIONS.')
        self.assertNotIn('Excluded', ' '.join(p['text'] for p in paragraphs))

    def test_james_split_uses_whole_paragraphs_before_sentence_expansion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'james_varieties.txt'
            path.write_text('LECTURE I. RELIGION AND NEUROLOGY.\n' +
                '\n\n'.join(f'Word{i}. More{i}.' for i in range(10)) + '\n\nPOSTSCRIPT.\n')
            data = prepare_corpus(path, 100)
            self.assertEqual([len(data[k]) for k in ('train', 'validation', 'test')], [16, 2, 2])
            self.assertNotIn('word8', data['vocabulary'])
            self.assertEqual(data['metadata']['validation']['first_paragraph_index'], 8)
            self.assertEqual([p['split'] for p in data['paragraphs']], ['train']*8+['validation', 'test'])

    def test_initials_and_typographic_hyphens_are_consistent(self):
        self.assertEqual(split_sentences('W. K. Clifford writes. Faith‐state matters.'),
                         [['w', 'k', 'clifford', 'writes'], ['faith-state', 'matters']])
        self.assertEqual(context_tokens('W. K. Clifford writes'), ['w', 'k', 'clifford', 'writes'])

if __name__ == '__main__':
    unittest.main()
