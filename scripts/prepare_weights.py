"""Prepare the local TSV checkpoint layout from versioned metadata and verified downloads."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'weights'
MODEL = ROOT / 'checkpoints' / 'TSV-28M'
MODES = ('auto', 'local')


def file_hash(path):
    with path.open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def install_file(target, source, *, checksum=None, url=False):
    """Only replace a cached file after its entire replacement has been verified."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', suffix='.part', delete=False) as out:
        temp = Path(out.name)
    try:
        with temp.open('wb') as out, (urlopen(source, timeout=120) if url else source.open('rb')) as stream:
            copyfileobj(stream, out, length=1024 * 1024)
        if checksum and file_hash(temp) != checksum:
            raise ValueError(f'SHA-256 mismatch for {target.name}; check the source and revision.')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def prepare_metadata():
    # These small tracked files also allow train.py/test.py --help to work offline.
    files = {name: MODEL / name for name in ('model_source.json', 'release.json', 'source.json')}
    files['backbone_config.json'] = MODEL / 'pretrained/config.json'
    files['LICENSE'] = MODEL / 'pretrained/LICENSE'
    for name, target in files.items():
        source = CONFIG / name
        if not target.is_file() or source.read_bytes() != target.read_bytes():
            install_file(target, source)


def fetch(target, expected, mode, url, *, hint):
    if target.is_file() and file_hash(target) == expected:
        return
    if mode == 'local' or not url:
        raise FileNotFoundError(f'Missing or incorrect file: {target}. {hint}')
    if urlsplit(url).scheme != 'https':
        raise ValueError(f'Expected an HTTPS download URL for {target.name}')
    print(f'Downloading {target.name} -> {target}', flush=True)
    install_file(target, url, checksum=expected, url=True)


def prepare_backbone(mode):
    info = json.loads((CONFIG / 'model_source.json').read_text(encoding='utf-8'))
    base = f"https://huggingface.co/{info['repo_id']}/resolve/{info['revision']}"
    for name, spec in info['files'].items():
        fetch(MODEL / info['local_directory'] / name, spec['sha256'], mode, f'{base}/{name}',
              hint='Use --pretrained auto to obtain the pinned official Timestral files.')


def prepare_checkpoint(path, mode):
    if path.resolve() == (MODEL / 'best.pt').resolve():
        info = json.loads((CONFIG / 'release.json').read_text(encoding='utf-8'))
        url = os.environ.get('TSV_RELEASE_URL') or info.get('download_url')
        fetch(path.resolve(), info['checkpoint_sha256'], mode, url,
              hint='Supply the team best.pt locally or use auto with TSV_RELEASE_URL. '
                   'Alternatively pass --checkpoint for a locally trained TSV model.')
    elif not path.is_file():
        raise FileNotFoundError(f'TSV checkpoint not found: {path}')


def add_mode(parser):
    parser.add_argument('--pretrained', choices=MODES, default='auto',
                        help='auto downloads missing weights; local uses already prepared files')


def prepare_entrypoint(*, training):
    # Handle acquisition outside the fingerprinted forecast/training source. Existing
    # local training checkpoints retain their exact code identity on this release.
    options = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_mode(options)
    parsed, forwarded = options.parse_known_args()
    help_only = any(arg in ('-h', '--help') for arg in forwarded)
    prepare_metadata()
    if help_only:
        print(options.format_help().split('\n', 1)[-1])
    else:
        if not training:
            from tsv.inputs import default_checkpoint
            paths = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            paths.add_argument('--checkpoint', type=Path)
            paths.add_argument('--output', type=Path, default=ROOT / 'outputs')
            chosen, _ = paths.parse_known_args(forwarded)
            prepare_checkpoint(chosen.checkpoint or default_checkpoint(chosen.output), parsed.pretrained)
        prepare_backbone(parsed.pretrained)
    sys.argv[1:] = forwarded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_mode(parser)
    parser.add_argument('--with-release', action='store_true', help='Also prepare the fine-tuned team best.pt')
    args = parser.parse_args()
    prepare_metadata()
    if args.with_release:
        prepare_checkpoint(MODEL / 'best.pt', args.pretrained)
    prepare_backbone(args.pretrained)
    print(f'Verified checkpoint files in {MODEL}')
