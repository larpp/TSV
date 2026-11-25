"""Generate the three Run-level partitions directly under the requested data directory."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset.build_mt_nit_csv_dataset import build, verify


ROOT = Path(__file__).resolve().parents[1]


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('The output directory must be outside the raw input tree')
    output.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in ('train', 'valid', 'test', 'manifest.json') if (output / name).exists()]
    if occupied:
        raise FileExistsError(f'Existing dataset in {output} ({", ".join(occupied)}). Choose another --output.')
    with TemporaryDirectory(dir=output, prefix='.prepare-') as stage:
        dataset = build(source, Path(stage))
        manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
        reference = json.loads((ROOT / 'config' / 'reference_split.json').read_text(encoding='utf-8'))
        current = {r['source_sha256']: [r['csv_path'], manifest['files'][r['csv_path']]]
                   for r in manifest['selected']}
        if current.keys() == reference.keys():
            if current != reference:
                raise ValueError('Reference split membership or processed CSV bytes differ')
            print(f'Reproduced all {len(current)} reference Run IDs, partitions and CSV SHA-256 hashes.')
        # Publish the manifest last, after every file it names exists.
        files = sorted(dataset.iterdir(), key=lambda path: path.name == 'manifest.json')
        for path in files:
            if (output / path.name).exists():
                raise FileExistsError(output / path.name)
        moved = []
        try:
            for path in files:
                path.rename(output / path.name)
                moved.append(path)
        except BaseException:
            for path in reversed(moved):
                (output / path.name).rename(path)
            raise
    print(f'Dataset ready: {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'data/mt_nit/data_complete')
    p.add_argument('--output', type=Path, default=ROOT / 'data')
    p.add_argument('--verify', action='store_true', help='Recheck the published CSVs against the raw inputs')
    args = p.parse_args()
    if args.verify:
        verify(args.output, args.source, reproduce=True)
    else:
        prepare(args.source, args.output)
