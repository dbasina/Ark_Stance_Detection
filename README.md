# Ark Stance Detection

Small research project for stance detection and aspect-masked ABSA experiments.

## Repo layout

- `Dataset/` — data files (csv, json, etc.).
- `Narratives/` — narrative files and tweet id lists.
- `dataset_analysis.py` — small script to inspect the dataset and generate `Dataset/unique_words.txt`.
- `dataset.py` — PyTorch Dataset wrapper `maskedABSA_Dataset` for the weakly labeled race dataset.
- `Ark_Aspect_Masking/` — model code and helpers for image/text models (if present).

## Quick start

1. Install minimal dependencies (example):

```bash
pip install pandas torch timm transformers
```

2. Inspect dataset and generate the list of unique words:

```bash
python3 dataset_analysis.py
```

This will print a sample and write `Dataset/unique_words.txt` (now stored as a Python list literal).

3. Use the PyTorch dataset in `dataset.py`:

```python
from dataset import maskedABSA_Dataset
train_ds = maskedABSA_Dataset("Dataset/weakly_labeled_race_dataset.csv", split='train')
```

`maskedABSA_Dataset` supports optional `split` (`'train'|'val'|'test'`) and `annotation_percent` to subsample.

## Notes

- `Dataset/unique_words.txt` is written as a Python list literal by `dataset_analysis.py`.
- Large binary/model files should be tracked with Git LFS if you plan to push them to a remote.

## License

Add your preferred license here.
