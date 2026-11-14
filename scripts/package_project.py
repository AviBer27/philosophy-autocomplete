"""Build the portable submission ZIP without environments or temporary files."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'output' / 'philosophy_autocomplete_project.zip'
FILES = [
    '.gitignore', 'README.md', 'requirements.txt', 'requirements-report.txt',
    'predict.py', 'experiment.py', 'Philosophy_Autocomplete.ipynb',
    'data/SOURCE.md', 'data/corpus_metadata.json',
    'data/raw/james_varieties.txt', 'data/raw/apollo11_technical.html',
    'output/results.json', 'output/results.csv', 'output/corpus_metadata.json',
    'output/parsed_paragraphs.json', 'output/selected_model.json',
    'output/experiment_specification.json', 'output/validation_frozen.json',
    'output/pdf/project_report.pdf',
]
DIRECTORIES = ['autocomplete', 'tests', 'scripts', 'output/models']


def main():
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(p for p in (ROOT / name).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc', '.DS_Store'))
    paths = sorted(set(paths))
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Build the project artifacts first: ' + ', '.join(missing))
    DESTINATION.parent.mkdir(exist_ok=True)
    with ZipFile(DESTINATION, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'{DESTINATION} ({DESTINATION.stat().st_size:,} bytes; {len(paths)} files)')


if __name__ == '__main__':
    main()
