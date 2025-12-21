# MVTec Anomaly Detection

This project implements and compares anomaly detection methods on the **MVTec AD dataset**
under a **CPU-only (laptop) environment**.

The goal is to analyze the trade-offs between:
- learning-based methods,
- statistical methods,
- and memory-based SOTA methods,

in terms of **performance (AUROC)** and **runtime efficiency**.

---

## Methods Implemented

- **AutoEncoder (AE)**  
  - Learning-based baseline  
  - Trained only on normal samples  

- **PaDiM**  
  - Statistical anomaly detection using multivariate Gaussian modeling  
  - Uses Mahalanobis distance on pretrained ResNet features  

- **PatchCore**  
  - Memory-based SOTA anomaly detection method  
  - Stores normal patch features and detects anomalies via kNN distance  

All methods share the same backbone and evaluation pipeline for fair comparison.

---

## Dataset

- **MVTec AD**
- Dataset is **not included** in this repository.
- Expected directory structure:

## Environment

- Python 3.11
- PyTorch (CPU)
- torchvision
- scikit-learn
