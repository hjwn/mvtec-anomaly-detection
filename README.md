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

## Directory
```text
data/mvtec_ad/
  bottle/
    train/
      good/*.png
    test/
      good/*.png
      broken_large/*.png
      broken_small/*.png
    ground_truth/
      broken_large/*_mask.png
      broken_small/*_mask.png
  capsule/
  cable/
  carpet/
  grid/
  hazelnut/
  leather/
  metal_nut/
  pill/
  screw/
  tile/
  toothbrush/
  transistor/
  wood/
  zipper/
```

## Environment

- Python 3.11
- PyTorch (CPU)
- torchvision
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Outputs
For each category, results are saved as:
```perl
outputs/<category>/
  ae/
  padim/
  patchcore/
```

Each method produces:

*_img.png : input image

*_gt.png : ground truth mask (white = defect)

*_map.png : anomaly heatmap (grayscale)

*_overlay.png : input image with heatmap overlay

A summary table is also generated:
outputs/summary.csv

## Evaluation Metrics
- Image AUROC
AUROC computed using a single anomaly score per image

- Pixel AUROC
AUROC computed using pixel-level anomaly heatmaps and ground-truth masks

---

# MVTec 이상 탐지 (Anomaly Detection)

본 프로젝트는 **MVTec AD 데이터셋**을 사용하여 
CPU 기반(노트북 환경)에서 이상 탐지(결함 탐지) 기법들을 구현하고 비교한다.

학습 기반 방법, 통계 기반 방법, 그리고 최신 메모리 기반 SOTA 기법을  
**동일한 실험 파이프라인**에서 비교하는 것을 목표로 한다.

---

## 구현된 방법

- **AutoEncoder (AE)**  
  - 학습 기반 이상 탐지 베이스라인  
  - 정상 데이터만을 사용하여 재구성 오류로 이상 판단  

- **PaDiM**  
  - 통계 기반 이상 탐지 기법  
  - 사전학습된 ResNet 특징에 대해 위치별 다변량 가우시안 분포를 모델링  
  - Mahalanobis distance를 이상 점수로 사용  

- **PatchCore**  
  - 메모리 기반 SOTA 이상 탐지 기법  
  - 정상 패치 특징을 저장하고 kNN 거리 기반으로 이상 판단  

모든 방법은 **동일한 backbone과 평가 기준**을 사용하여 공정하게 비교된다.

---

## 데이터셋

- **MVTec Anomaly Detection (MVTec AD)**
- 데이터셋은 저장소에 포함되어 있지 않다.
- 기대하는 디렉토리 구조는 다음과 같다:


---

## 실행 환경

- Python 3.11
- PyTorch (CPU)
- torchvision
- scikit-learn

의존성 설치:

```bash
pip install -r requirements.txt
```