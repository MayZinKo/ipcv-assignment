"""
IPCV Assignment - Thresholding Comparison Experiment
Professor: Leonardo Antonio Noriega Williams
Due: 3rd July 2026

Compares three thresholding methods on the Zalcode Bubble Answer Dataset:
  - Otsu's thresholding
  - Adaptive Gaussian thresholding
  - Sauvola thresholding

Classification: Support Vector Machine (SVM) with multiple image features
Classes: filled, default, crossed, invalid

HOW TO RUN:
  1. Activate your environment:
       conda activate ipcv_env

  2. Run:
       python ipcv_experiment.py

  3. Results are saved to a folder called: ipcv_results/
"""

import os
import sys
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_sauvola
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = "BubbleSheetDataset"
OUTPUT_DIR   = "ipcv_results"
IMAGE_SIZE   = (64, 64)
CLASSES      = ["crossed", "default", "filled", "invalid"]
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_images_from_csv(split_folder):
    """
    Load images and labels from a split folder.
    Labels are read from _classes.csv inside that folder.
    """
    images, labels = [], []
    split_path = Path(split_folder)
    csv_path   = split_path / "_classes.csv"

    if not split_path.exists():
        print(f"  WARNING: folder not found: {split_path}, skipping.")
        return images, labels

    if not csv_path.exists():
        print(f"  WARNING: _classes.csv not found in {split_path}, skipping.")
        return images, labels

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row

        for row in reader:
            if len(row) < 5:
                continue

            filename   = ",".join(row[:-4]).strip()
            class_vals = row[-4:]

            try:
                class_vals = [int(v.strip()) for v in class_vals]
            except ValueError:
                continue

            if sum(class_vals) != 1:
                continue

            class_index = class_vals.index(1)
            label       = CLASSES[class_index]

            img_path = split_path / filename
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label)

    return images, labels


def load_all_images(dataset_path):
    """Load all images from train, test, and valid splits."""
    all_images, all_labels = [], []

    for split in ["train", "test", "valid"]:
        split_folder = os.path.join(dataset_path, split)
        imgs, lbls   = load_images_from_csv(split_folder)
        all_images.extend(imgs)
        all_labels.extend(lbls)
        print(f"  {split:<6}: {len(imgs)} images loaded")

    print(f"  {'─'*30}")
    print(f"  {'Total':<6}: {len(all_images)} images loaded")
    print()
    print("  Class distribution:")
    for cls in CLASSES:
        count = all_labels.count(cls)
        print(f"    {cls:<10}: {count} images")

    return all_images, all_labels


# ─────────────────────────────────────────────
#  THRESHOLDING METHODS
# ─────────────────────────────────────────────

def apply_otsu(image):
    """Apply Otsu's global thresholding."""
    image = cv2.GaussianBlur(image, (3, 3), 1.0)
    _, binary = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_adaptive_gaussian(image):
    """Apply Adaptive Gaussian thresholding."""
    image = cv2.GaussianBlur(image, (3, 3), 1.0)
    binary = cv2.adaptiveThreshold(image, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary


def apply_sauvola(image):
    """Apply Sauvola thresholding."""
    image = cv2.GaussianBlur(image, (3, 3), 1.0)
    thresh = threshold_sauvola(image, window_size=25)
    binary = (image > thresh).astype(np.uint8) * 255
    return binary


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(binary_image):
    """
    Extract multiple features from a thresholded bubble image.

    Features:
    1. Fill ratio        - proportion of dark pixels
    2. Edge density      - how many edges exist (detects X marks)
    3. Pixel variance    - how uneven the darkness is
    4. H symmetry        - horizontal symmetry score
    5. V symmetry        - vertical symmetry score
    6. Centre fill ratio - fill ratio of the centre region only
    """
    h, w = binary_image.shape

    # 1. Fill ratio — proportion of dark pixels
    dark_pixels = np.sum(binary_image < 128)
    fill_ratio  = dark_pixels / binary_image.size

    # 2. Edge density — edges reveal X marks and crosses
    edges        = cv2.Canny(binary_image, 50, 150)
    edge_density = np.sum(edges > 0) / binary_image.size

    # 3. Pixel variance — high variance = uneven marks
    variance = np.var(binary_image.astype(float)) / (255.0 ** 2)

    # 4. Horizontal symmetry — compare top half vs bottom half
    top    = binary_image[:h//2, :]
    bottom = binary_image[h//2:, :]
    bottom_flipped = np.flip(bottom, axis=0)
    min_h  = min(top.shape[0], bottom_flipped.shape[0])
    h_sym  = 1.0 - np.mean(np.abs(top[:min_h].astype(float) -
                                   bottom_flipped[:min_h].astype(float))) / 255.0

    # 5. Vertical symmetry — compare left half vs right half
    left  = binary_image[:, :w//2]
    right = binary_image[:, w//2:]
    right_flipped = np.flip(right, axis=1)
    min_w = min(left.shape[1], right_flipped.shape[1])
    v_sym = 1.0 - np.mean(np.abs(left[:, :min_w].astype(float) -
                                   right_flipped[:, :min_w].astype(float))) / 255.0

    # 6. Centre fill ratio — focus on the inner 50% of the image
    cy1, cy2 = h//4, 3*h//4
    cx1, cx2 = w//4, 3*w//4
    centre       = binary_image[cy1:cy2, cx1:cx2]
    centre_fill  = np.sum(centre < 128) / centre.size

    return np.array([fill_ratio, edge_density, variance,
                     h_sym, v_sym, centre_fill])


# ─────────────────────────────────────────────
#  SVM CLASSIFIER
# ─────────────────────────────────────────────

def build_feature_matrix(images, threshold_fn):
    """Extract features from all images using one thresholding method."""
    """
        example return output
        image 1  → [0.02, 0.08, 0.05, 0.95, 0.96, 0.01]
        image 2  → [0.75, 0.03, 0.12, 0.92, 0.91, 0.80]
        image 3  → [0.40, 0.11, 0.09, 0.72, 0.68, 0.35]
        ...
        image 656→ [0.01, 0.02, 0.03, 0.97, 0.98, 0.01]
    """
    features = []
    for img in images:
        binary  = threshold_fn(img)
        feats   = extract_features(binary)
        features.append(feats)
    return np.array(features)


def train_and_evaluate(images, true_labels, threshold_fn, method_name):
    """
    images — all 656 images, 
    true_labels — correct answers for each image,
    threshold_fn — the thresholding function (Otsu, Adaptive Gaussian, or Sauvola)
    method_name — just the method name for printing
    
    Train an SVM classifier on features extracted using one
    thresholding method, then evaluate it using cross-validation.
    """
    print(f"  Extracting features using {method_name}...")
    X = build_feature_matrix(images, threshold_fn)
    y = np.array(true_labels)
    # X row 1 → [0.02, 0.08, 0.05, 0.95, 0.96, 0.01], y row 1 → "default"

    # Normalise features
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM
    print(f"  Training SVM classifier...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)

    # Cross-validation — 5 folds for reliable evaluation
    print(f"  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring="f1_weighted")
    print(f"  CV F1 scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean CV F1  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Final fit on all data for confusion matrix
    svm.fit(X_scaled, y)
    predictions = svm.predict(X_scaled)

    return predictions, cv_scores


# ─────────────────────────────────────────────
#  METRICS AND VISUALISATION
# ─────────────────────────────────────────────

def compute_metrics(true_labels, predictions, method_name):
    """Compute and print classification metrics."""
    acc  = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions,
                           average="weighted", zero_division=0)
    rec  = recall_score(true_labels, predictions,
                        average="weighted", zero_division=0)
    f1   = f1_score(true_labels, predictions,
                    average="weighted", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  Method: {method_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print()
    print(classification_report(true_labels, predictions,
                                 labels=CLASSES, zero_division=0))

    return {"method": method_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(true_labels, predictions, method_name, output_dir):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, predictions, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix — {method_name}", fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fname = os.path.join(output_dir,
                         f"confusion_{method_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison_bar(results, output_dir):
    """Save a grouped bar chart comparing all three methods."""
    methods   = [r["method"]    for r in results]
    accuracy  = [r["accuracy"]  for r in results]
    f1        = [r["f1"]        for r in results]
    precision = [r["precision"] for r in results]
    recall    = [r["recall"]    for r in results]

    x     = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, accuracy,  width, label="Accuracy",  color="#2E5090")
    ax.bar(x - 0.5*width, precision, width, label="Precision", color="#4C8FD6")
    ax.bar(x + 0.5*width, recall,    width, label="Recall",    color="#76B7E8")
    ax.bar(x + 1.5*width, f1,        width, label="F1 Score",  color="#A8D4F5")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Thresholding Method Comparison — All Metrics", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    fname = os.path.join(output_dir, "method_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def save_results_txt(results, output_dir):
    """Save a plain text summary of all results."""
    fname = os.path.join(output_dir, "results_summary.txt")
    with open(fname, "w") as f:
        f.write("IPCV ASSIGNMENT — EXPERIMENT RESULTS\n")
        f.write("="*55 + "\n")
        f.write("Dataset    : Zalcode Bubble Answer Dataset\n")
        f.write(f"Classes    : {', '.join(CLASSES)}\n")
        f.write("Classifier : Support Vector Machine (SVM, RBF kernel)\n")
        f.write("Features   : fill ratio, edge density, variance,\n")
        f.write("             h-symmetry, v-symmetry, centre fill ratio\n\n")
        f.write(f"{'Method':<30} {'Accuracy':>10} {'Precision':>10} "
                f"{'Recall':>10} {'F1 Score':>10}\n")
        f.write("-"*72 + "\n")
        for r in results:
            f.write(f"{r['method']:<30} {r['accuracy']:>10.4f} "
                    f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
                    f"{r['f1']:>10.4f}\n")
        f.write("\n")
        best = max(results, key=lambda x: x["f1"])
        f.write(f"Best method by F1 Score: {best['method']} "
                f"(F1 = {best['f1']:.4f})\n")
    print(f"  Saved: {fname}")


def show_sample_images(images, true_labels, output_dir):
    """Save sample outputs showing original + all three thresholded versions."""
    methods = [
        ("Otsu's",            apply_otsu),
        ("Adaptive Gaussian", apply_adaptive_gaussian),
        ("Sauvola",           apply_sauvola),
    ]

    samples = {}
    for img, lbl in zip(images, true_labels):
        if lbl not in samples:
            samples[lbl] = img
        if len(samples) == len(CLASSES):
            break

    fig, axes = plt.subplots(len(samples), 4,
                              figsize=(12, 3 * len(samples)))
    col_titles = ["Original"] + [m[0] for m in methods]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, (class_name, img) in enumerate(samples.items()):
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_ylabel(class_name, fontsize=10, rotation=0,
                                 labelpad=55, va="center")
        axes[row, 0].axis("off")
        for col, (_, fn) in enumerate(methods, start=1):
            binary = fn(img)
            axes[row, col].imshow(binary, cmap="gray")
            axes[row, col].axis("off")

    plt.suptitle("Sample Bubble Images — Three Thresholding Methods",
                 fontsize=13)
    plt.tight_layout()
    fname = os.path.join(output_dir, "sample_outputs.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  IPCV ASSIGNMENT — THRESHOLDING + SVM EXPERIMENT")
    print("="*60)

    if not os.path.exists(DATASET_PATH):
        print(f"\n  ERROR: Dataset path not found: {DATASET_PATH}")
        print("  Make sure you are running this script from the same")
        print("  folder that contains the BubbleSheetDataset folder.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Results will be saved to: ./{OUTPUT_DIR}/")

    # [1] Load images
    print("\n[1] Loading images (train + test + valid)...")
    images, true_labels = load_all_images(DATASET_PATH)

    if len(images) == 0:
        print("\n  ERROR: No images loaded. Check your dataset folder.")
        sys.exit(1)

    # [2] Run all three methods
    print("\n[2] Running experiments (Thresholding + SVM)...")
    methods = [
        ("Otsu's Thresholding",  apply_otsu),
        ("Adaptive Gaussian",    apply_adaptive_gaussian),
        ("Sauvola Thresholding", apply_sauvola),
    ]

    all_results     = []
    all_predictions = {}

    for name, fn in methods:
        print(f"\n  ── {name} ──")
        preds, cv_scores = train_and_evaluate(images, true_labels, fn, name)
        metrics          = compute_metrics(true_labels, preds, name)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"]  = cv_scores.std()
        all_predictions[name] = preds
        all_results.append(metrics)

    # [3] Save results
    print("\n[3] Saving results...")
    save_results_txt(all_results, OUTPUT_DIR)

    # [4] Generate visualisations
    print("\n[4] Generating visualisations...")
    for name, fn in methods:
        plot_confusion_matrix(true_labels, all_predictions[name],
                              name, OUTPUT_DIR)
    plot_comparison_bar(all_results, OUTPUT_DIR)
    show_sample_images(images, true_labels, OUTPUT_DIR)

    # Final summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  {'Method':<30} {'Accuracy':>10} {'F1 Score':>10} {'CV F1':>10}")
    print("  " + "-"*62)
    for r in all_results:
        print(f"  {r['method']:<30} {r['accuracy']*100:>9.2f}% "
              f"{r['f1']:>10.4f} {r['cv_mean']:>10.4f}")
    best = max(all_results, key=lambda x: x["f1"])
    print(f"\n  Best method : {best['method']}")
    print(f"  Best F1     : {best['f1']:.4f}")
    print(f"\n  All files saved to: ./{OUTPUT_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()