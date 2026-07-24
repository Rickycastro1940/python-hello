"""Dogs vs Cats image classification — Steps 2–5.

Step 2: visualize samples and prepare 224x224 train/test generators.
Step 3: build the VGG16-style ANN and compile it.
Step 4: optimize with ModelCheckpoint + EarlyStopping; predict on test.
Step 5: persist the best model under models/.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
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
HISTORY_PNG = FIGURES / "training_history.png"
PREDICTIONS_PNG = FIGURES / "test_predictions.png"
# Tutorial/solution checkpoint name (Keras 3 native format)
CHECKPOINT_PATH = MODELS_DIR / "vgg16_1.keras"

# VGG-style input size from the tutorial architecture
IMG_SIZE = (224, 224)
TEST_FRACTION = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 16
# Match the tutorial solution fit() settings
EPOCHS = 3
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 10
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 3


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


def compile_model(model: Sequential) -> Sequential:
    """Compile with Adam + categorical cross-entropy (tutorial Step 3)."""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks(checkpoint_path: Path) -> list:
    """Create ModelCheckpoint + EarlyStopping (tutorial Step 4)."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )
    early = EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOP_PATIENCE,
        verbose=1,
        mode="auto",
        restore_best_weights=False,
    )
    return [checkpoint, early]


def optimize_with_callbacks(model: Sequential, trdata, tsdata, checkpoint_path: Path):
    """Train with checkpoint/early-stopping callbacks (fit replaces fit_generator)."""
    callbacks = make_callbacks(checkpoint_path)
    print(
        f"Step 4 training: epochs={EPOCHS}, "
        f"steps_per_epoch={STEPS_PER_EPOCH}, validation_steps={VALIDATION_STEPS}"
    )
    # Modern Keras: model.fit(...) — fit_generator() was removed.
    history = model.fit(
        trdata,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=tsdata,
        validation_steps=VALIDATION_STEPS,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def plot_history(history, out_path: Path) -> None:
    """Plot training/validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def index_to_label(class_indices: dict, index: int) -> str:
    """Map softmax argmax index back to class name."""
    inv = {v: k for k, v in class_indices.items()}
    return inv[index]


def predict_on_test_set(model, tsdata, out_path: Path, n_show: int = 9) -> float:
    """Load-time predictions on the test generator; save a sample grid."""
    print("\n=== Step 4: Predictions on the test set ===")
    probs = model.predict(tsdata, verbose=1)
    pred_idx = np.argmax(probs, axis=1)
    true_idx = tsdata.classes
    accuracy = float(np.mean(pred_idx == true_idx))
    print(f"Test-set prediction accuracy: {accuracy:.2%}")

    # Sample grid: first n_show images from the generator filenames
    filepaths = list(tsdata.filepaths)
    n_show = min(n_show, len(filepaths))
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle("Best-model predictions on test images", fontsize=14)
    for ax, i in zip(axes.ravel(), range(n_show)):
        path = filepaths[i]
        img = load_img(path, target_size=IMG_SIZE)
        true_name = index_to_label(tsdata.class_indices, int(true_idx[i]))
        pred_name = index_to_label(tsdata.class_indices, int(pred_idx[i]))
        conf = float(np.max(probs[i]))
        ax.imshow(img)
        ax.set_title(f"true={true_name}\npred={pred_name} ({conf:.2f})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return accuracy


def predict_single_image(model, image_path: Path, class_indices: dict) -> str:
    """Predict one image the way the tutorial solution does."""
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    output = model.predict(arr, verbose=0)
    # class_indices is alphabetical with flow_from_directory: {'cat': 0, 'dog': 1}
    cat_i = class_indices["cat"]
    dog_i = class_indices["dog"]
    label = "cat" if output[0][cat_i] > output[0][dog_i] else "dog"
    print(
        f"Single-image prediction for {image_path.name}: {label} "
        f"(cat={output[0][cat_i]:.4f}, dog={output[0][dog_i]:.4f})"
    )
    return label


def save_model_to_folder(model, models_dir: Path = MODELS_DIR) -> Path:
    """Step 5: store the trained model in the models/ folder."""
    models_dir.mkdir(parents=True, exist_ok=True)
    # Match the tutorial/solution filename stem (vgg16_1); use native .keras format
    out_path = models_dir / "vgg16_1.keras"
    model.save(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Step 5: model saved to {out_path} ({size_mb:.1f} MB)")
    return out_path


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

    print("\n=== Step 3: Build and compile the VGG16-style ANN ===")
    model = build_vgg16_like_model()
    compile_model(model)
    model.summary()

    print("\n=== Step 4: Optimize with ModelCheckpoint + EarlyStopping ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    history = optimize_with_callbacks(model, trdata, tsdata, CHECKPOINT_PATH)
    plot_history(history, HISTORY_PNG)

    if not CHECKPOINT_PATH.exists():
        # Fallback if val_accuracy never improved enough to write a checkpoint
        model.save(CHECKPOINT_PATH)
        print(f"No checkpoint file yet — saved final weights to {CHECKPOINT_PATH}")

    print(f"\nLoading best model from {CHECKPOINT_PATH}")
    best_model = load_model(CHECKPOINT_PATH)

    test_loss, test_acc = best_model.evaluate(tsdata, verbose=1)
    print(f"Best-model test loss: {test_loss:.4f}")
    print(f"Best-model test accuracy: {test_acc:.2%}")

    predict_on_test_set(best_model, tsdata, PREDICTIONS_PNG)

    # One concrete example image (first test file), like the solution notebook
    sample_path = Path(tsdata.filepaths[0])
    predict_single_image(best_model, sample_path, tsdata.class_indices)

    print("\n=== Step 5: Save the model ===")
    saved_path = save_model_to_folder(best_model, MODELS_DIR)
    print(f"Model is ready in the models folder: {saved_path}")


if __name__ == "__main__":
    main()
