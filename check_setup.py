"""Environment and ESC-50 dataset smoke test. No model training."""
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parent
print("Repository:", ROOT)
print("\nDependencies:")
required = ("numpy", "pandas", "torch", "torchaudio", "librosa", "soundfile", "sklearn", "lime", "ipykernel")
missing = []
for package in required:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "installed")
        print(f"  {package}: {version}")
    except Exception as error:
        print(f"  {package}: FAILED ({type(error).__name__}: {error})")
        missing.append(package)

if not missing:
    import torch
    print("\nPyTorch CUDA available:", torch.cuda.is_available())

print("\nDataset:")
candidates = sorted((ROOT / "data").glob("*/meta/esc50.csv"))
if not candidates:
    print("  NOT FOUND. Extract the official ESC-50 ZIP under data/.")
    print("  Expected: data/ESC-50-master/meta/esc50.csv")
    raise SystemExit(1)

import pandas as pd
csv_path = candidates[0]
df = pd.read_csv(csv_path)
audio_dir = csv_path.parent.parent / "audio"
required_cols = {"filename", "fold", "target", "category", "src_file"}
assert required_cols.issubset(df.columns), "ESC-50 metadata columns missing"
missing_files = [name for name in df["filename"] if not (audio_dir / name).is_file()]
print("  Metadata:", csv_path)
print("  Clips:", len(df), "(expected: 2000)")
print("  Classes:", df["target"].nunique(), "(expected: 50)")
print("  Folds:", df["fold"].value_counts().sort_index().to_dict())
print("  Missing WAVs:", len(missing_files))
assert len(df) == 2000 and df["target"].nunique() == 50
assert set(df["fold"].unique()) == {1,2,3,4,5}
assert not missing_files, f"Missing audio file: {missing_files[0]}"

# Report source IDs that appear in more than one fold in the metadata.
src_overlap = df.groupby("src_file")["fold"].nunique()
overlapping_sources = sorted(src_overlap[src_overlap > 1].index.tolist())
if overlapping_sources:
    print("  WARNING: source IDs appear in multiple folds:", overlapping_sources)
else:
    print("  Original source recordings are fold-disjoint: YES")
print("  Train / validation / test clips:", [len(df[df["fold"].isin(f)]) for f in ([1,2,3],[4],[5])])
print("\nSetup check passed.")
