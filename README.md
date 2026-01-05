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

PaDiM and PatchCore share the same ResNet backbone and evaluation pipeline.
AE uses its own encoder-decoder and is trained on unnormalized inputs, while
PaDiM/PatchCore use ImageNet-normalized inputs for the pretrained backbone.

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
AUROC computed using a single anomaly score per image (max over heatmap)

- Pixel AUROC
AUROC computed using pixel-level anomaly heatmaps and ground-truth masks

---

# MVTec 이상 탐지 (Anomaly Detection)

본 프로젝트는 **MVTec AD 데이터셋**을 사용하여 
CPU 기반(노트북 환경)에서 이상 탐지(결함 탐지) 기법들을 구현하고 비교합니다.

학습 기반 방법, 통계 기반 방법, 그리고 최신 메모리 기반 SOTA 기법을  
**동일한 실험 파이프라인**에서 비교하는 것을 목표로 합니다.

---

## 비교 모델

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

PaDiM과 PatchCore는 동일한 ResNet backbone과 평가 기준을 사용합니다.
AE는 별도의 인코더-디코더를 사용하며 입력 정규화는 하지 않았습니다.
PaDiM/PatchCore는 사전학습 backbone을 위해 ImageNet 정규화를 사용하였습니다.

## 데이터셋

- **MVTec Anomaly Detection (MVTec AD)**
- 데이터셋은 repository에 포함되어 있지 않습니다.
- 기대하는 디렉토리 구조는 다음과 같습니다:
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

## 결과 (Outputs)
각 카테고리 별로 결과는 다음과 같이 저장됩니다.
```perl
outputs/<category>/
  ae/
  padim/
  patchcore/
```

각 방법은 아래 파일들을 생성합니다.

### *_img.png : 입력 이미지

### *_gt.png : 정답 마스크 (흰색 = 결함)

### *_map.png : 이상 점수 히트맵 (그레이스케일)

### *_overlay.png : 입력 이미지 + 히트맵 오버레이

요약 테이블도 함께 저장됩니다.
outputs/summary.csv

## 평가 지표
- Image AUROC  
  이미지별 이상 점수(heatmap의 최대값)를 기준으로 AUROC를 계산합니다.

- Pixel AUROC  
  픽셀 단위 이상 점수와 정답 마스크를 사용해 AUROC를 계산합니다.
