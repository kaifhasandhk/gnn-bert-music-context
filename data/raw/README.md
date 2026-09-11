# Raw Audio Datasets

Due to GitHub file size limits, raw audio files (`.mp3`) and full CSV annotations are excluded from version control via `.gitignore`.

## Dataset Source & Expected Structure

This project uses the **MagnaTagATune** dataset (available via Kaggle at `/kaggle/input/datasets/yalumusic/magnatagatune/`).

Expected directory layout for local reproduction:

```text
data/raw/
├── annotations_final.csv
└── MagnaTagATune/
    ├── 0/    # mp3 audio clips ]
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    ├── 7/
    ├── 8/
    ├── 9/
    ├── a/
    ├── b/
    ├── c/
    ├── d/
    ├── e/
    └── f/
