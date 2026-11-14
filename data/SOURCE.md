# Corpus provenance

The main experiment uses William James, *The Varieties of Religious Experience:
A Study in Human Nature*, Project Gutenberg eBook 621.

- Author: William James (1842-1910).
- Catalog: https://www.gutenberg.org/ebooks/621
- Official catalog-linked plain text: https://www.gutenberg.org/cache/epub/621/pg621.txt
- Downloaded: 2026-11-14.
- Raw bytes: 1171335.
- SHA-256: `bcfadbaeb796fdd8a721909409d27e42a81a9beed814b60d708a9f4559712bf2`.
- Source encoding: UTF-8; the complete Gutenberg notice and license remain in
  `raw/james_varieties.txt`. The official catalog labels the work public domain
  in the USA.

The model input contains only the twenty main lectures, beginning at the exact
uppercase Lecture I heading and ending before the uppercase POSTSCRIPT heading.
The preface, contents, postscript, index, endnotes and Gutenberg wrappers are
excluded. Lecture headings, decorative separators and 359 attached numeric
footnote references are removed. The lecture text includes James's quotations;
it is not limited to sentences personally authored by James. Spaced inline
numbered lists and the text's legitimate parenthesized numerical citation remain.

The parser keeps 1,171 source paragraphs. It assigns whole paragraphs in source
order to train/validation/test (936/117/118), then splits sentences within each
assigned paragraph. No paragraph spans subsets. Lecture headings and source-position metadata are not model features. Quoted passages are ordinary
model input. `output/parsed_paragraphs.json` records the source lines, lecture,
text, order and split of every retained paragraph.

Typography is normalized to NFC, curly apostrophes to straight apostrophes, and
Unicode hyphens to ASCII hyphens. Wrapped lines are joined within paragraphs.
Tokenization lowercases words while retaining internal apostrophes/hyphens,
alphanumeric forms, numbers and decimals. Initials, common abbreviations and
decimals are protected before heuristic `.`, `!`, `?` sentence splitting. This is
a simple reproducible rule rather than a linguistic sentence segmenter.

The experiment reads the local source without network access. The exact parsing
and train-only vocabulary construction are in `autocomplete/data.py`.

## Optional Apollo corpus and archived first experiment

The previous domain experiment remains reproducible with `--corpus apollo`.
Its original results and report are preserved in `output/archive/apollo_v1`.

NASA, *Apollo 11 Technical Air-to-Ground Voice Transcription* (GOSS NET 1), raw
uncorrected OCR HTML courtesy Thomas Schwagmeier.

- Catalog: https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11trans.html
- Download: https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11transcript_tec.html
- Downloaded: 2026-11-12.
- Raw bytes: 1273321.
- SHA-256: `daf141a2c9030b0ca1e8748962af7a4cbfe227ed324f5c5ca085f063db271b3a`.
- Original encoding: legacy Windows-1252, explicitly decoded by the parser.
- The official catalog states that these raw transcripts were provided by NASA
  to the press during the mission, have not been corrected, and no copyright
  is asserted. The separately copyrighted corrected Lunar Surface Journal is
  excluded.

The optional parser extracts whole timestamped transmissions, excludes
introduction/page/tape/station/revolution headings and other source annotations,
and retains time/speaker fields only as provenance. OCR wording remains
uncorrected. Whole transmissions are split by source order before sentence
expansion. The archived run retains its original preprocessing/results; the
current shared sentence heuristic additionally protects academic initials.
