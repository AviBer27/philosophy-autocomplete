"""Build the portable submission ZIP without environments or temporary files."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Use this PDF as the report inside the ZIP (for a private submission copy).')
    parser.add_argument('--output', type=Path, default=DESTINATION, help='Destination ZIP file.')
    args = parser.parse_args()
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(p for p in (ROOT / name).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc', '.DS_Store'))
    paths = sorted(set(paths))
    # Browser uploads commonly omit hidden files. The ignore file is optional
    # and is not needed to run or package an otherwise complete checkout.
    paths = [p for p in paths if p.name != '.gitignore' or p.is_file()]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()
               and not (args.report and p == ROOT / 'output/pdf/project_report.pdf')]
    if args.report and not args.report.is_file():
        missing.append(str(args.report))
    if missing:
        raise FileNotFoundError('Build the project artifacts first: ' + ', '.join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            source = args.report if args.report and path == ROOT / 'output/pdf/project_report.pdf' else path
            archive.write(source, path.relative_to(ROOT).as_posix())
    print(f'{args.output} ({args.output.stat().st_size:,} bytes; {len(paths)} files)')


if __name__ == '__main__':
    main()
