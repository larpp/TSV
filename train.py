"""TSV-28M train entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tsv.train import main
from scripts.prepare_weights import prepare_entrypoint

if __name__ == "__main__":
    prepare_entrypoint(training=True)
    main()
