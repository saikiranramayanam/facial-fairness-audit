# Facial Verification Fairness Audit

## Project Overview

This project implements a facial verification system and performs a comprehensive fairness audit across demographic groups including gender, age, and skin tone.

The system uses a deep learning embedding model trained with Triplet Loss to perform face verification. The project additionally evaluates demographic fairness using False Accept Rate (FAR) and False Reject Rate (FRR), applies a mitigation strategy, and compares fairness-performance trade-offs.

---

# Objectives

- Build a facial verification system
- Generate facial embeddings using deep learning
- Perform demographic fairness auditing
- Measure FAR and FRR across subgroups
- Implement bias mitigation
- Evaluate fairness vs accuracy trade-offs
- Produce reproducible results using Docker

---

# Dataset

Dataset used:

- UTKFace Dataset

The dataset provides:
- Age
- Gender
- Ethnicity

These attributes were used to define demographic groups.

---

# Demographic Groups

## Gender
- Male
- Female

## Age Bins
- 0–19
- 20–39
- 40–59
- 60+

## Skin Tone Groups
- Light
- Medium
- Dark

Definitions are stored in:

```text
results/demographics.json
```

---

# Model Architecture

The project uses:

- ResNet18 backbone
- 128-dimensional embedding layer
- Triplet Margin Loss

The model learns facial similarity embeddings instead of performing classification.

---

# Training Pipeline

1. Load facial images
2. Generate triplets:
   - Anchor
   - Positive
   - Negative
3. Train embedding model using Triplet Loss
4. Save trained model artifact

Model artifacts:

```text
artifacts/model.pt
artifacts/model_mitigated.pt
```

---

# Fairness Audit

The fairness audit evaluates:

- False Accept Rate (FAR)
- False Reject Rate (FRR)

Metrics are computed for:
- Overall population
- Individual demographic subgroups

Audit reports:

```text
results/initial_audit.json
results/mitigated_audit.json
```

---

# Bias Mitigation Strategy

The project implements:

## Balanced Batch Sampling

Training data was balanced across:
- Gender
- Race groups

This mitigation strategy aimed to reduce demographic imbalance during training.

---

# Results

## Initial Model
- Accuracy: 0.5000

## Mitigated Model
- Accuracy: 0.5042

The mitigation strategy slightly improved fairness-aware performance while producing more differentiated subgroup metrics.

---

# Project Structure

```text
artifacts/
data/
results/
submission/
src/

Dockerfile
docker-compose.yml
README.md
requirements.txt
```

---

# Running the Project

## Docker Setup

Build and run:

```bash
docker-compose up --build
```

---

# Training

```bash
python src/train.py
```

---

# Initial Audit

```bash
python src/audit.py
```

---

# Bias Analysis

```bash
python src/analyze_bias.py
```

---

# Mitigation

```bash
python src/mitigation.py
```

---

# Mitigated Audit

```bash
python src/mitigated_audit.py
```

---

# Metrics Comparison

```bash
python src/compare_metrics.py
```

---

# Ethical Considerations

Although mitigation techniques were applied, facial verification systems still present important ethical concerns:

- Demographic performance disparities
- Dataset imbalance
- Potential misuse in surveillance systems
- False positive security risks
- Unequal real-world impacts

The system should therefore be deployed cautiously in high-stakes environments.

---

# Final Recommendation

This system demonstrates a complete fairness auditing pipeline suitable for educational and research purposes.

However, due to remaining demographic disparities and moderate verification accuracy, the model is not recommended for high-stakes production deployment without significantly larger datasets and more advanced debiasing techniques.