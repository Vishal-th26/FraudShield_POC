# FraudShield — Hybrid Fraud Detection POC

FraudShield is a **hybrid fraud detection prototype** designed to identify suspicious financial transactions using a layered machine learning approach.  
It combines a strong supervised baseline with adaptive online anomaly detection and drift monitoring to better handle evolving fraud behavior.

This repository contains the **core detection engine (POC)** and preprocessing pipeline that power the larger FraudShield concept.

---

## Overview

Fraud detection is not just a classification problem — it is a moving target.

FraudShield addresses this by using a **multi-layer hybrid detection architecture**:

1. **LightGBM** as the primary supervised fraud scorer  
2. **Half-Space Trees (River)** for online unsupervised anomaly detection  
3. **Online Meta-Learner (River Logistic Regression)** to learn the best blend of signals over time  
4. **ADWIN Drift Detector** to monitor distribution shifts and flag when retraining may be needed

This design is aimed at combining:
- **accuracy** from supervised learning,
- **adaptability** from online anomaly detection,
- and **robustness** against concept drift in real-world fraud streams.

---

## Hybrid Detection Engine

The detection engine follows a three-part intelligence flow:

### 1) LightGBM — Primary Scorer
A supervised model trained on labeled fraud data to produce the main fraud probability score.  
It is optimized for tabular fraud data and serves as the first-line detector.

### 2) Half-Space Trees (River)
An online unsupervised anomaly detector that updates continuously as transactions arrive.  
It helps catch unusual patterns that may not yet be represented in labeled data.

### 3) Online Meta-Learner (River LR)
A lightweight online learner that blends the outputs of the primary model and anomaly detector.  
This layer improves decision quality by adapting to feedback and changing fraud patterns.

### 4) ADWIN Drift Detector
A drift monitoring component that watches the score distribution over time.  
When behavior changes significantly, it can trigger a retraining recommendation.

---

## Dataset

This project uses the **IEEE-CIS Fraud Detection dataset** for model development and evaluation.

The dataset provides a realistic fraud detection setting with:
- highly imbalanced classes,
- noisy real-world transaction patterns,
- and a large number of engineered and categorical features.

---

## Why This Approach

Fraud patterns evolve constantly. A single model can become stale quickly.

FraudShield is designed to reduce that weakness by combining:
- a **high-performing supervised classifier**,
- an **online anomaly detector**,
- a **meta-learning blend layer**,
- and **drift monitoring** for adaptation.

This makes the system more suitable for practical fraud analytics than a single static model.

---

## Features

- End-to-end preprocessing pipeline
- Hybrid fraud scoring engine
- Supervised + unsupervised detection combination
- Online learning layer for adaptive score fusion
- Drift detection for changing fraud distributions
- Modular architecture for future expansion
- Prototype demo support for presentation and validation

---

## Tech Stack

- **Python**
- **Pandas / NumPy**
- **LightGBM**
- **River**
- **Scikit-learn**
- **Joblib**
- **FastAPI** *(if applicable)*
- **React** *(for demo frontend, if applicable)*

---

## Repository Structure

```bash
fraudshield/
├── data/
│   ├── raw/
│   ├── processed/
│   └── ...
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── ...
├── src/
│   ├── preprocessing.py
│   ├── hybrid_model.py
│   ├── inference.py
│   └── ...
├── demo/
│   └── prototype_video.mp4
├── README.md
└── requirements.txt





Demo

🎥 Prototype Demo Video:
https://drive.google.com/file/d/14blBkConaRPCM-KzTO2iG4hFi0rpyWVv/view?usp=sharing

The demo presents the FraudShield concept as an end-to-end fraud intelligence workflow built around the core POC engine in this repository.


This repository represents the core proof-of-concept for FraudShield.

Current scope:

•preprocessing pipeline
•hybrid detection engine
•prototype-level validation

Planned expansion:

•retrieval-based fraud case intelligence
•explainability layer
•structured compliance report generation
•full dashboard integration
