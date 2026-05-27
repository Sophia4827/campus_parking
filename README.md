# Forecasting Campus Parking: Predicting Binghamton University Lot Occupancy Using YOLO

This repo contains the code, results, model evaluation outputs, and final report for a computer vision project that detects and classifies parking space occupancy from lot images using YOLO. The project also includes a Figma prototype of a mobile app designed to help Binghamton University commuters make informed parking decisions in real time.

Here are the different components:

- [YOLO Testing Notebook](./YOLO%20Testing.ipynb): Full pipeline including data preprocessing, annotation conversion, model training, hyperparameter tuning, and evaluation
- [Lot Occupancy Predictions](./Lot_Occupancy_PredictionsNEW.csv): Final model output — lot occupancy at each timestamp for which an image was collected
- [Final Project Report](./Project_Report.pdf): Full written report covering methodology, results, mobile application design, and scalability framework
- [Figma Mobile App Prototype](https://pitch-squad-47294237.figma.site)

## Overview

Parking congestion at Binghamton University is a persistent problem for commuters. The university's Transportation and Parking Services (TAPS) currently tracks availability through manual counts by student workers, a process that is labor-intensive, error-prone, and operates on a one-hour lag. Security cameras are already installed across campus lots, making an automated, image-based solution both technically feasible and cost-effective.

This project addresses that gap by building a computer vision pipeline using **YOLO26s** (Ultralytics' latest small-variant object detection model) trained on **PKLot**, a publicly available university parking lot dataset, as a stand-in for Binghamton's camera infrastructure.

## Results

| Data Split | Precision | Recall | F1 | mAP50 |
|------------|-----------|--------|----|-------|
| Train | 0.994 | 0.993 | 0.994 | 0.994 |
| Test | 0.904 | 0.931 | 0.917 | 0.876 |

| Class | Precision | Recall | F1 | mAP50 |
|-------|-----------|--------|----|-------|
| Empty | 0.927 | 0.870 | 0.898 | 0.873 |
| Occupied | 0.880 | 0.993 | 0.923 | 0.933 |

- Detected **255,528 spaces** across **6,010 test images**
- Occupied space detection achieved nearly perfect recall (0.993) — the model is more likely to flag a space as occupied than to miss it, which is the preferred outcome for availability reporting

### Confusion Matrix
![Confusion Matrix](./YOLO26/confusion_matrix_normalized.png)

### Precision-Recall Curve
![PR Curve](./YOLO26/BoxPR_curve.png)

## Methods

**Dataset:** PKLot — 12,417 images with 695,899 individually labeled parking spaces, collected at a Brazilian university with usage patterns similar to Binghamton's (predictable high-traffic periods tied to class schedules). Data spans September 2012 – April 2013 across varying lighting, weather, and occupancy conditions.

**Data Preprocessing:**
- Restructured train/test/validation splits to replicate TAPS's semesterly workflow (one semester of data used to predict the next)
- Converted PKLot JSON annotations to YOLO-compatible per-image label files with normalized center coordinates

**Model:** YOLO26s — the small variant of Ultralytics' YOLO26, chosen for its balance of speed and accuracy suitable for real-time deployment without excessive hardware requirements

**Training & Tuning:**
- Initial training: 14 epochs on 5,867 images (434,768 bounding boxes)
- Secondary training with hyperparameter tuning: image augmentation (scale, rotate, translate, blur), learning rate decay to prevent overfitting

## Mobile Application

The application interface was designed in Figma and centers on simplicity for commuters. Key features include:

- **Aerial map view** of selected parking lots with color-coded occupancy overlays (green / orange / red)
- **Section breakdown panel** with exact available space counts and occupancy progress bars
- **Day and time filtering** to look up anticipated availability ahead of a commute
- **Peak hours chart** showing how occupancy typically evolves throughout the day
- **"Right Now" button** that auto-sets filters to the user's current moment

View the prototype: [Figma Mobile App Prototype](https://pitch-squad-47294237.figma.site)

## Scalability

The report includes a full framework for deploying this system on Binghamton University's existing camera infrastructure, covering scope selection (all commuter lots vs. high-capacity vs. high-traffic), camera installation guidelines, model retraining on BU-specific data, and application development priorities including bMobi integration.

## Requirements

- Python 3.x
- Jupyter Notebook

## Dependencies

- ultralytics
- torch
- pandas
- numpy
- matplotlib
- opencv-python

## Note on Model Weights & Data

Model weight files (`.pt`) and raw image datasets are not included due to file size constraints. The PKLot dataset is publicly available [here](https://web.inf.ufpr.br/vri/databases/parking-lot-database/).

## Authors

Sophia Huang · Lydia Wyble  
Binghamton University MS Data Analytics · Spring 2026
