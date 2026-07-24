# Dogs vs Cats Image Classifier

4Geeks [image-classifier-project-tutorial](https://github.com/4GeeksAcademy/image-classifier-project-tutorial) — Steps 1–3.

## What is implemented

1. **Step 1** — Download/unzip `dogs-vs-cats` under `data/raw/` (see commands below).
2. **Step 2** — Preview first 9 dogs/cats; build train/test class folders; `ImageDataGenerator` pipelines at **224×224** (VGG input).
3. **Step 3** — Build the tutorial **VGG16-style** `Sequential` CNN and compile with Adam.
4. **Step 4** — Optimize with `ModelCheckpoint` + `EarlyStopping`, reload the best weights, predict on the test set.
5. **Step 5** — Persist the best model under `models/vgg16_1.keras`.

## Architecture (Step 3)

Matches the syllabus VGG16-style stack:

- Conv blocks: 64 → 128 → 256 → 512 → 512 with `MaxPool2D`
- Head: `Flatten` → `Dense(4096)` → `Dense(4096)` → `Dense(2, softmax)`
- Optimizer: Adam `1e-4`, loss: `categorical_crossentropy`

## Run

```bash
cd image-classifier-project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Step 1: dataset
mkdir -p data/raw && cd data/raw
curl -L -o dogs-vs-cats.zip \
  "https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip"
unzip dogs-vs-cats.zip
cd ../..

# Steps 2–3
python main.py
```

CPU note: full VGG (~134M params) is heavy. Defaults use `EPOCHS=3`, `STEPS_PER_EPOCH=100`, `VALIDATION_STEPS=25` so a smoke-run finishes on CPU. Increase those for a real fit.

## Outputs

| Path | Description |
|------|-------------|
| `figures/dogs_preview.png` / `cats_preview.png` | Sample grids |
| `figures/training_history.png` | Acc/loss curves |
| `figures/test_predictions.png` | Sample test predictions |
| `models/vgg16_1.keras` | Best checkpoint (Step 4/5) |

`fit_generator` is deprecated/removed in modern Keras — Step 4 uses `model.fit(..., callbacks=[...])` with the same checkpoint/early-stopping objects the syllabus describes.
