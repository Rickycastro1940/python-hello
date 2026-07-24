"""Dogs vs Cats image classification — Steps 2–3.

Step 2: visualize samples and prepare 224x224 train/test generators.
Step 3: build a VGG16-style CNN, train it, and measure performance.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

# Paths
ROOT = Path(__file__).resolve().parent
RAW_TRAIN = ROOT / "data" / "raw" / "dogs-vs-cats" / "train"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS_DIR = ROOT / "models"
DATASET_NPZ = PROCESSED / "dogs_vs_cats_224x224.npz"
HISTORY_PNG = FIGURES / "training_history.png"

# VGG-style input size from the tutorial architecture
IMG_SIZE = (224, 224)
TEST_FRACTION = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 3
# Limit steps so CPU training finishes in a reasonable time on full VGG
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 25


def list_labeled_images(train_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return sorted dog and cat image paths from the flat train folder."""
    dogs = sorted(train_dir.glob("dog.*.jpg"))
    cats = sorted(train_dir.glob("cat.*.jpg"))
    if not dogs or not cats:
        raise FileNotFoundError(
            f"Expected dog.*.jpg / cat.*.jpg under {train_dir}. "
            "Run Step 1 (download + unzip) first."
        )
    return dogs, cats


def plot_first_nine(image_paths: list[Path], title: str, out_path: Path) -> None:
    """Load and print the first nine pictures in a single figure."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(title, fontsize=14)

    for ax, path in zip(axes.ravel(), image_paths[:9]):
        img = load_img(path)
        ax.imshow(img)
        ax.set_title(f"{path.name}\n{img.size[0]}x{img.size[1]}")
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def build_train_test_folders(
    dogs: list[Path],
    cats: list[Path],
    processed_dir: Path,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[Path, Path]:
    """Create train/test directories with cat/ and dog/ class subfolders."""
    rng = random.Random(seed)
    dogs_shuffled = dogs[:]
    cats_shuffled = cats[:]
    rng.shuffle(dogs_shuffled)
    rng.shuffle(cats_shuffled)

    def split(paths: list[Path]) -> tuple[list[Path], list[Path]]:
        n_test = int(len(paths) * test_fraction)
        return paths[n_test:], paths[:n_test]

    train_dogs, test_dogs = split(dogs_shuffled)
    train_cats, test_cats = split(cats_shuffled)

    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"

    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    mapping = {
        train_dir / "dog": train_dogs,
        train_dir / "cat": train_cats,
        test_dir / "dog": test_dogs,
        test_dir / "cat": test_cats,
    }

    for dest_dir, paths in mapping.items():
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            link = dest_dir / src.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(src.resolve(), link)

    print(
        f"Train/test folders ready under {processed_dir} "
        f"(train dogs={len(train_dogs)}, cats={len(train_cats)}; "
        f"test dogs={len(test_dogs)}, cats={len(test_cats)})"
    )
    return train_dir, test_dir


def make_generators(train_dir: Path, test_dir: Path):
    """Create ImageDataGenerator objects for training and test data."""
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # categorical because the VGG head uses Dense(2, softmax)
    trdata = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )
    tsdata = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    print(f"class_indices: {trdata.class_indices}")
    return trdata, tsdata


def build_vgg16_like_model() -> Sequential:
    """Build the VGG16-style CNN from the tutorial (Step 3)."""
    model = Sequential()
    model.add(
        Conv2D(
            input_shape=(224, 224, 3),
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    return model


def plot_history(history, out_path: Path) -> None:
    """Plot training/validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def prepare_data_if_needed() -> tuple:
    """Ensure preview figures + train/test generators exist."""
    dogs, cats = list_labeled_images(RAW_TRAIN)
    print(f"Found {len(dogs)} dog images and {len(cats)} cat images")

    FIGURES.mkdir(parents=True, exist_ok=True)
    if not (FIGURES / "dogs_preview.png").exists():
        plot_first_nine(
            dogs, "First 9 dog images (original sizes)", FIGURES / "dogs_preview.png"
        )
    if not (FIGURES / "cats_preview.png").exists():
        plot_first_nine(
            cats, "First 9 cat images (original sizes)", FIGURES / "cats_preview.png"
        )

    train_dir = PROCESSED / "train"
    test_dir = PROCESSED / "test"
    if not (train_dir / "dog").exists() or not (test_dir / "cat").exists():
        PROCESSED.mkdir(parents=True, exist_ok=True)
        train_dir, test_dir = build_train_test_folders(dogs, cats, PROCESSED)

    trdata, tsdata = make_generators(train_dir, test_dir)
    return trdata, tsdata


def main() -> None:
    print("=== Step 2: Data ready for the network ===")
    trdata, tsdata = prepare_data_if_needed()
    batch_x, batch_y = next(iter(trdata))
    print(f"trdata batch: X={batch_x.shape}, y={batch_y.shape}")

    print("\n=== Step 3: Build a VGG16-style ANN ===")
    model = build_vgg16_like_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print(
        f"\nTraining for {EPOCHS} epoch(s) "
        f"(steps_per_epoch={STEPS_PER_EPOCH}, validation_steps={VALIDATION_STEPS})..."
    )
    history = model.fit(
        trdata,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=tsdata,
        validation_steps=VALIDATION_STEPS,
        epochs=EPOCHS,
        verbose=1,
    )

    print("\n=== Evaluation on test generator ===")
    test_loss, test_acc = model.evaluate(tsdata, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2%}")

    plot_history(history, HISTORY_PNG)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / "vgg16_dogs_cats.keras"
    model.save(weights_path)
    print(f"Saved model checkpoint to {weights_path}")


if __name__ == "__main__":
    main()
