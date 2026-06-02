# IPCV Assignment — Automated Bubble Sheet Grading

**Student:** May Zin Ko  
**Module:** Image Processing and Computer Vision (IPCV)  

---

## Project Overview

This project compares three image thresholding methods for automated 
bubble sheet grading using a Support Vector Machine (SVM) classifier.

**Methods compared:**
- Otsu's Thresholding
- Adaptive Gaussian Thresholding
- Sauvola Thresholding

---

## Folder Structure

ipcv-assignment/
│
├── BubbleSheetDataset/       # Dataset from Kaggle (Zalcode)
│   ├── train/                # 486 training images + _classes.csv
│   ├── test/                 # 39 test images + _classes.csv
│   └── valid/                # 127 validation images + _classes.csv
│
├── ipcv_results/             # Output folder (generated after running)
│   ├── results_summary.txt   # Accuracy, Precision, Recall, F1 scores
│   ├── method_comparison.png # Bar chart comparing all methods
│   ├── sample_outputs.png    # Sample thresholded bubble images
│   ├── confusion_Otsu's_Thresholding.png
│   ├── confusion_Adaptive_Gaussian.png
│   └── confusion_Sauvola_Thresholding.png
│
├── ipcv_experiment.py        # Main experiment code
└── README.md                 # This file
---

## How to Run

**Step 1 — Install required libraries:**
```bash
pip install opencv-python scikit-image scikit-learn matplotlib seaborn numpy
```

**Step 2 — Run the experiment:**
```bash
python ipcv_experiment.py
```

**Step 3 — Check results:**  
All output files will be saved in the `ipcv_results/` folder.

---

## Results Summary

| Method | Accuracy | F1 Score |
|---|---|---|
| Otsu's Thresholding | 95.86% | 0.9585 |
| Sauvola Thresholding | 92.94% | 0.9294 |
| Adaptive Gaussian | 81.13% | 0.8067 |

**Best method: Otsu's Thresholding (F1 = 0.9585)**

---

## Dataset

Zalcode (2023) *Bubble Answer Dataset*. Kaggle.  
Available at: https://www.kaggle.com/datasets/zalcode/bubble-answer-dataset

---

## Requirements

- Python 3.x
- opencv-python
- scikit-image
- scikit-learn
- matplotlib
- seaborn
- numpy



