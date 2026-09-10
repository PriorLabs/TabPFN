#  Copyright (c) Prior Labs GmbH 2026.
"""Predict from a picture and a few tabular fields, together.

An image goes into TabPFN as a base64 string in a DataFrame cell, in a column
declared through `image_features_indices`. Each such column is replaced by the
PCA-reduced CLS embedding of a frozen DINOv3 ViT-S/16 before the tabular model sees
it. The script fits three models, on the tabular fields alone, on the picture alone
and on both, and prints each score.

The data is a MulTaBench task (https://arxiv.org/abs/2605.10616), one of three, named
on the command line; the default is the smallest. The archives download without a
Kaggle account. The encoder's weights are gated: accept the license at
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and run
`hf auth login` once.

    pip install "tabpfn[image]" kagglehub
    python tabpfn_with_images.py [amazon-bestseller | glaucoma-smdg | hateful-meme]
"""

import base64
import json
import sys
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor

DATASETS = {
    # Log price of an Amazon best seller from its photo, rank and ratings. 55 MB.
    "amazon-bestseller": "chico89/multabench-amazon-bestseller",
    # Glaucoma present, absent or suspected, from a fundus photo and patient data. 3 GB.
    "glaucoma-smdg": "chico89/multabench-glaucoma-smdg",
    # Whether a meme is hateful, from its picture and a text embedding. 3.2 GB.
    "hateful-meme": "chico89/multabench-hateful-meme",
}
dataset = sys.argv[1] if len(sys.argv) > 1 else "amazon-bestseller"

root = Path(kagglehub.dataset_download(DATASETS[dataset]))
meta = json.loads((root / "metadata.json").read_text())
data = pd.read_csv(root / "data.csv")
y = data.pop(meta["target"])
data["image"] = [
    base64.b64encode((root / path).read_bytes()).decode("ascii")
    for path in data.pop(meta["image_col"])
]
regression = meta["task_type"] == "reg"
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.25, stratify=None if regression else y, random_state=0
)

tabular = list(data.columns.drop("image"))
for name, columns in [
    ("tabular only", tabular),
    ("picture only", ["image"]),
    ("tabular + picture", [*tabular, "image"]),
]:
    image_indices = [i for i, column in enumerate(columns) if column == "image"]
    if regression:
        model = TabPFNRegressor(image_features_indices=image_indices)
        model.fit(X_train[columns], y_train)
        score = r2_score(y_test, model.predict(X_test[columns]))
    else:
        model = TabPFNClassifier(image_features_indices=image_indices)
        model.fit(X_train[columns], y_train)
        proba = model.predict_proba(X_test[columns])
        score = roc_auc_score(
            y_test,
            proba[:, 1] if proba.shape[1] == 2 else proba,
            multi_class="ovr",
            average="macro",
        )
    print(f"{'R2' if regression else 'ROC AUC'}, {name}: {score:.3f}")
