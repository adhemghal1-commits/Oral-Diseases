"""
Step 2: Model Training & Comparative Analysis Pipeline
======================================================
Trains and compares 4 architectures on the Oral Disease dataset:
  1. Custom CNN (Built from scratch)
  2. MobileNetV2 (Pretrained on ImageNet - Transfer Learning)
  3. ResNet50 (Pretrained on ImageNet - Transfer Learning)
  4. VGG16 (Pretrained on ImageNet - Transfer Learning)

Includes:
  - Hyperparameter tuning for the top performing architecture
  - Multi-class evaluation (Accuracy, Loss, Precision, Recall, F1-score)
  - Confusion matrices and comparative visualization plots
  - Export of best model for Flask deployment
"""

import os
import json
import time
from pathlib import Path
import numpy as np

# Suppress TF oneDNN and verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import keras
from keras import layers, models, callbacks, optimizers, utils

# ── Paths & Parameters ────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\adhem\Downloads\Oral Disease")
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
NUM_CLASSES = 6
BASE_EPOCHS = 3
STEPS_PER_EPOCH = 35
VAL_STEPS = 12
TEST_STEPS = 20

print("=" * 65)
print("  Oral Disease Deep Learning Pipeline (Keras 3 + TensorFlow)")
print("=" * 65)
print(f"Base Directory:  {BASE_DIR}")
print(f"Image Size:      {IMG_SIZE}")
print(f"Batch Size:      {BATCH_SIZE}")

# ── Load Datasets using modern keras.utils ─────────────────────────────────────
print("\n[1/5] Loading and caching dataset splits...")

raw_train_ds = utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42,
)

raw_val_ds = utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

raw_test_ds = utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

CLASS_NAMES = raw_train_ds.class_names
print(f"Detected Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

# Save class names for Flask
with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)

# Optimization: prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = raw_test_ds.prefetch(buffer_size=AUTOTUNE)

# Extract ground truth test labels for evaluation
y_true_list = []
for _, labels in raw_test_ds:
    y_true_list.append(np.argmax(labels.numpy(), axis=1))
y_test_true = np.concatenate(y_true_list, axis=0)


# ── Data Augmentation Pipeline ────────────────────────────────────────────────
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ],
    name="data_augmentation",
)


# ── Model Architectures ───────────────────────────────────────────────────────

def build_custom_cnn():
    """Custom Convolutional Neural Network built from scratch."""
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="Custom_CNN")


def build_mobilenetv2():
    """MobileNetV2 Transfer Learning Backbone."""
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2")
    return model, base_model


def build_resnet50():
    """ResNet50 Transfer Learning Backbone."""
    base_model = keras.applications.ResNet50(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet50")
    return model, base_model


def build_vgg16():
    """VGG16 Transfer Learning Backbone."""
    base_model = keras.applications.VGG16(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="VGG16")
    return model, base_model


# ── Plotting Helpers ──────────────────────────────────────────────────────────

def plot_dataset_distribution():
    """Plots and saves the dataset sample counts across classes and splits."""
    splits = ["train", "val", "test"]
    data = {s: {} for s in splits}
    for s in splits:
        split_path = DATASET_DIR / s
        for cls_dir in sorted(split_path.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.*")))
                data[s][cls_dir.name] = count

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black", linewidth=0.6)
    ax.set_title("Oral Diseases Dataset - Class Distribution Across Splits", fontsize=13, fontweight="bold")
    ax.set_xlabel("Disease Category", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "dataset_distribution.png", dpi=200)
    plt.close()
    print("  ✓ Saved plots/dataset_distribution.png")


def plot_confusion_matrix_figure(cm, model_name):
    """Generates and saves a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(PLOT_DIR / f"confusion_matrix_{safe_name}.png", dpi=200)
    plt.close()
    print(f"  ✓ Saved plots/confusion_matrix_{safe_name}.png")


def plot_metrics_summary(results_dict, histories_dict):
    """Generates overall accuracy comparison, loss curves, and per-class F1 charts."""
    # 1. Bar Chart Comparison
    models_list = list(results_dict.keys())
    accuracies = [results_dict[m]["test_accuracy"] * 100 for m in models_list]
    losses = [results_dict[m]["test_loss"] for m in models_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    bars = ax1.bar(models_list, accuracies, color=palette[:len(models_list)], edgecolor="black", linewidth=0.7)
    ax1.set_title("Test Accuracy Comparison Across Models", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.2f}%", ha="center", fontweight="bold")
    ax1.set_xticklabels(models_list, rotation=20, ha="right")

    bars2 = ax2.bar(models_list, losses, color=palette[:len(models_list)], edgecolor="black", linewidth=0.7)
    ax2.set_title("Test Loss Comparison (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontweight="bold")
    ax2.set_xticklabels(models_list, rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "accuracy_comparison.png", dpi=200)
    plt.close()
    print("  ✓ Saved plots/accuracy_comparison.png")

    # 2. Training Comparison Curves
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(15, 5))
    for m in models_list:
        if m in histories_dict:
            hist = histories_dict[m]
            ax_acc.plot(hist["accuracy"], label=f"{m} (Train)", linestyle="-")
            ax_acc.plot(hist["val_accuracy"], label=f"{m} (Val)", linestyle="--")
            ax_loss.plot(hist["loss"], label=f"{m} (Train)", linestyle="-")
            ax_loss.plot(hist["val_loss"], label=f"{m} (Val)", linestyle="--")

    ax_acc.set_title("Model Training & Validation Accuracy Curves", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.grid(True, linestyle="--", alpha=0.4)

    ax_loss.set_title("Model Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_comparison.png", dpi=200)
    plt.close()
    print("  ✓ Saved plots/training_comparison.png")

    # 3. Per-Class F1 Score Comparison
    f1_records = []
    for m in models_list:
        report = results_dict[m]["classification_report"]
        for cls in CLASS_NAMES:
            if cls in report:
                f1_records.append({
                    "Model": m,
                    "Class": cls,
                    "F1_Score": report[cls]["f1-score"],
                })
    f1_df = pd.DataFrame(f1_records)
    plt.figure(figsize=(13, 6))
    sns.barplot(data=f1_df, x="Class", y="F1_Score", hue="Model", palette="tab10")
    plt.title("Per-Class F1-Score Breakdown by Model", fontsize=13, fontweight="bold")
    plt.xlabel("Oral Condition", fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "per_class_f1.png", dpi=200)
    plt.close()
    print("  ✓ Saved plots/per_class_f1.png")


# ── Training & Evaluation Routine ─────────────────────────────────────────────

def train_and_eval_model(model, name, lr=1e-3, epochs=BASE_EPOCHS):
    """Compiles, trains, and evaluates a given model on test split."""
    print(f"\n{'─'*65}")
    print(f"  Training Model: {name} (LR={lr}, Max Epochs={epochs})")
    print(f"{'─'*65}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    start_t = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        callbacks=cb,
        verbose=1,
    )
    elapsed = time.time() - start_t
    print(f"  Training time: {elapsed:.1f}s")

    # Save model file
    model_save_path = MODEL_DIR / f"{name}.keras"
    model.save(model_save_path)
    print(f"  Saved model artifact: {model_save_path.name}")

    # Evaluate on test set
    loss, acc = model.evaluate(test_ds, verbose=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_dict = classification_report(y_test_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_text = classification_report(y_test_true, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test_true, y_pred)

    print(f"\n  Test Accuracy: {acc * 100:.2f}% | Test Loss: {loss:.4f}")
    print(report_text)

    # Plot Confusion Matrix
    plot_confusion_matrix_figure(cm, name)

    return {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "training_time_sec": float(elapsed),
    }, history.history


# ── Hyperparameter Tuning Function ────────────────────────────────────────────

def run_hyperparameter_tuning(base_builder_fn, best_model_name):
    """
    Hyperparameter Tuning:
    Systematic tuning over learning rates and fine-tuning depths on the winning pretrained backbone.
    """
    print("\n" + "=" * 65)
    print(f"  Hyperparameter Tuning & Fine-Tuning: {best_model_name}")
    print("=" * 65)

    tuning_candidates = [
        {"lr": 1e-4, "unfreeze_pct": 0.20, "label": "Tune_LR_1e-4_Top20%"},
        {"lr": 5e-5, "unfreeze_pct": 0.35, "label": "Tune_LR_5e-5_Top35%"},
        {"lr": 2e-5, "unfreeze_pct": 0.50, "label": "Tune_LR_2e-5_Top50%"},
    ]

    best_tuning_acc = -1.0
    best_tuned_model = None
    best_config = None
    best_tuning_history = None

    for candidate in tuning_candidates:
        print(f"\n[Hyperparameter Trial] Learning Rate: {candidate['lr']} | Unfreeze: {candidate['unfreeze_pct']*100:.0f}%")
        
        # Build fresh model and backbone
        model, base_model = base_builder_fn()
        
        # Load weights from initial pretrained stage if available
        base_path = MODEL_DIR / f"{best_model_name}.keras"
        if base_path.exists():
            try:
                model.load_weights(base_path)
            except Exception:
                pass

        # Unfreeze top layers of backbone
        base_model.trainable = True
        num_layers = len(base_model.layers)
        freeze_idx = int(num_layers * (1.0 - candidate["unfreeze_pct"]))
        for l in base_model.layers[:freeze_idx]:
            l.trainable = False
        for l in base_model.layers[freeze_idx:]:
            l.trainable = True

        model.compile(
            optimizer=optimizers.Adam(learning_rate=candidate["lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        cb = [
            callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=2,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VAL_STEPS,
            callbacks=cb,
            verbose=1,
        )

        val_acc = max(history.history["val_accuracy"])
        print(f"  Trial Result -> Best Val Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_tuning_acc:
            best_tuning_acc = val_acc
            best_tuned_model = model
            best_config = candidate
            best_tuning_history = history.history

    print(f"\n  ★ Optimal Hyperparameters Selected: {best_config['label']}")
    print(f"    Best Validation Accuracy: {best_tuning_acc * 100:.2f}%")

    # Evaluate the fine-tuned tuned model
    loss, acc = best_tuned_model.evaluate(test_ds, verbose=0)
    y_pred = np.argmax(best_tuned_model.predict(test_ds, verbose=0), axis=1)
    report_dict = classification_report(y_test_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test_true, y_pred)

    name_label = f"{best_model_name}_FineTuned"
    plot_confusion_matrix_figure(cm, name_label)

    return best_tuned_model, {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "best_config": best_config,
    }, best_tuning_history, name_label


# ── Main Execution ────────────────────────────────────────────────────────────

def main():
    # Step A: Visualize dataset distribution
    plot_dataset_distribution()

    results_all = {}
    histories_all = {}

    # Step B: Train Custom CNN
    print("\n" + "=" * 65)
    print("  PHASE 1: Custom CNN Architecture (Scratch)")
    print("=" * 65)
    custom_cnn = build_custom_cnn()
    custom_res, custom_hist = train_and_eval_model(custom_cnn, "Custom_CNN", lr=1e-3, epochs=BASE_EPOCHS)
    results_all["Custom_CNN"] = custom_res
    histories_all["Custom_CNN"] = custom_hist

    # Step C: Train Pretrained Models
    print("\n" + "=" * 65)
    print("  PHASE 2: Pretrained Transfer Learning Backbones")
    print("=" * 65)

    pretrained_factories = {
        "MobileNetV2": build_mobilenetv2,
        "ResNet50": build_resnet50,
        "VGG16": build_vgg16,
    }

    pretrained_results = {}
    for name, factory_fn in pretrained_factories.items():
        model, _ = factory_fn()
        res, hist = train_and_eval_model(model, name, lr=5e-4, epochs=BASE_EPOCHS)
        results_all[name] = res
        histories_all[name] = hist
        pretrained_results[name] = res["test_accuracy"]

    # Step D: Select Best Pretrained Model for Hyperparameter Tuning
    best_pretrained_name = max(pretrained_results, key=pretrained_results.get)
    print(f"\nTop Pretrained Architecture for Tuning: {best_pretrained_name} "
          f"({pretrained_results[best_pretrained_name]*100:.2f}%)")

    best_factory = pretrained_factories[best_pretrained_name]
    best_tuned_model, tuned_res, tuned_hist, tuned_name = run_hyperparameter_tuning(
        best_factory, best_pretrained_name
    )
    results_all[tuned_name] = tuned_res
    histories_all[tuned_name] = tuned_hist

    # Step E: Save Final Deployment Model
    final_best_name = max(results_all, key=lambda k: results_all[k]["test_accuracy"])
    print(f"\n" + "=" * 65)
    print(f"  WINNER: {final_best_name} (Acc: {results_all[final_best_name]['test_accuracy']*100:.2f}%)")
    print("=" * 65)

    if final_best_name == tuned_name:
        best_tuned_model.save(MODEL_DIR / "best_model.keras")
    else:
        # Load and re-save winning model
        win_model = keras.models.load_model(MODEL_DIR / f"{final_best_name}.keras")
        win_model.save(MODEL_DIR / "best_model.keras")

    print(f"  ✓ Deployment model exported to: {MODEL_DIR / 'best_model.keras'}")

    # Step F: Generate Plots & Summaries
    plot_metrics_summary(results_all, histories_all)

    # Save complete JSON summary
    summary_path = MODEL_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"  ✓ Metrics saved to: {summary_path}")

    # Step G: Final Leaderboard
    print("\n" + "=" * 65)
    print("  FINAL COMPARATIVE LEADERBOARD")
    print("=" * 65)
    print(f"{'Rank':<6}{'Model Name':<28}{'Test Acc (%)':<15}{'Test Loss':<12}")
    print("─" * 65)
    sorted_models = sorted(results_all.items(), key=lambda kv: kv[1]["test_accuracy"], reverse=True)
    for idx, (m_name, metrics) in enumerate(sorted_models, 1):
        print(f"#{idx:<5}{m_name:<28}{metrics['test_accuracy']*100:>10.2f}%   {metrics['test_loss']:>10.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
