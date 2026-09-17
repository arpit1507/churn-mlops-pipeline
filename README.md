# 💳 Credit Card Fraud Detection: Production MLOps Pipeline

An end-to-end, reproducible Machine Learning Operations (MLOps) pipeline engineered to detect fraudulent credit card transactions under extreme class imbalance. 

This repository demonstrates industry-standard software engineering and ML systems practices: **modular pipeline architecture**, **strict Git feature-branching strategies**, **large-scale data versioning using DVC backed by AWS S3**, and cost-sensitive gradient-boosted modeling.

---

## 📌 Table of Contents
- [System Architecture](#system-architecture)
- [Dataset Overview & Inherent Challenges](#dataset-overview--inherent-challenges)
- [MLOps Engineering Principles](#mlops-engineering-principles)
  - [Why Git Feature Branching?](#why-git-feature-branching)
  - [DVC & AWS S3 Integration Architecture](#dvc--aws-s3-integration-architecture)
- [Pipeline Execution Stages](#pipeline-execution-stages)
- [Model Performance & Results Analysis](#model-performance--results-analysis)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
- [Reproducibility & Execution](#reproducibility--execution)

---

## 🏗 System Architecture

The pipeline decouples configuration, orchestration, data tracking, and execution into isolated stages:


```

[Kaggle / Remote Storage]
│
▼
┌─────────────────────────┐
│  Stage 01: Ingestion    │ ────► Raw CSV ────► Tracked via DVC ──► AWS S3 Bucket
└──────────┬──────────────┘
│
▼
┌─────────────────────────┐
│ Stage 02: Preparation   │ ────► Stratified Split + RobustScaler + SMOTE
└──────────┬──────────────┘       (Outputs train.parquet / test.parquet)
│
▼
┌─────────────────────────┐
│  Stage 03: Training     │ ────► Cost-Sensitive XGBoost (scale_pos_weight)
└──────────┬──────────────┘       (Outputs model.joblib + metrics.json)
│
▼
┌─────────────────────────┐
│ Stage 04: Evaluation    │ ────► AUPRC, ROC-AUC, Confusion Matrix, Threshold Tuning
└─────────────────────────┘

```

---

## 📊 Dataset Overview & Inherent Challenges

The pipeline processes the Kaggle Credit Card Fraud Detection benchmark:
* **Transactions:** 284,807 total transactions recorded over two days in September 2013.
* **Dimensionality:** 30 input features.
  * `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
  * `Amount`: Transaction monetary amount (heavily right-skewed, ranging from $0.00 to over $25,000).
  * `V1` to `V28`: Principal Component Analysis (PCA) transformed features due to confidentiality.
* **The Class Imbalance Challenge:**
  * **Normal (Class 0):** 284,315 transactions (~99.827%)
  * **Fraud (Class 1):** 492 transactions (~0.173%)

### Engineering Implications:
1. **The "Accuracy Paradox":** A naive dummy model predicting every record as legitimate achieves **99.83% accuracy** while catching 0 actual frauds. Accuracy is strictly prohibited as an evaluation metric.
2. **Heavy Outlier Sensitivity:** Features like `Amount` contain severe positive skews. Standard Z-score normalization collapses under extreme outliers; hence, **`RobustScaler`** (based on Interquartile Range, IQR) is applied.
3. **Data Leakage Mitigation:** Preprocessing parameters (medians, IQR scales, oversampling centroids) must be fitted strictly on training partitions and applied to testing partitions without target contamination.

---

## 🛠 MLOps Engineering Principles

### 1. Why Git Feature Branching?

In machine learning engineering, experimental chaos easily corrupts codebases. Rather than committing code, configs, and trial-and-error changes directly to `main`, this project employed a strict **Git Branching Strategy**:

* **Branch Structure:**
  * `main`: Production-ready, verified, deployable pipeline code.
  * `feature/data-ingestion`: Handles remote data pulls, directory creation, checksum validations, and DVC pointer setup.
  * `feature/data-preparation`: Implements deterministic data splitting, transformation pipelines, and SMOTE balancing.
  * `feature/model-training`: Focuses on model instantiation, hyperparameter definitions, and serialization.
* **Why this was used:**
  * **Stage Isolation:** Each pipeline stage is developed and tested as an independent micro-component. A bug in model training cannot corrupt the data ingestion verification logic.
  * **Clear Lineage & Auditability:** Merge points (such as `3506dd9` for Ingestion and `7382671` for Preparation) document the exact moment a capability entered the stable codebase.
  * **Conflict Control:** Multi-developer teams modifying `config.yaml` or `params.yaml` simultaneously can isolate structural schema additions and reconcile them cleanly via standard merge requests.

### 2. DVC & AWS S3 Integration Architecture

Git is engineered for lightweight source code diffs. Tracking 150MB+ raw datasets, 100MB+ Parquet files, or serialized binary models (`model.joblib`) directly inside Git causes:
* GitHub remote rejection errors (`RPC failed; HTTP 408 / size exceeded limit`).
* Repository bloat where cloning takes gigabytes of bandwidth.
* Zero native understanding of dataset diffs or pipeline stages.

#### The DVC + S3 Solution:
* **The Split-Plane Pattern:**
  * **Control Plane (Git):** Tracks code, configuration (`config.yaml`), and lightweight text pointers (`artifacts/*.dvc`, typically ~100 bytes containing MD5 content hashes).
  * **Data Plane (AWS S3):** Stores the actual binary blobs in an S3 bucket (`churn-mlops-dvc-storage-2026`) organized by hash keys.
* **Workflow:**
  ```bash
  # 1. Generate heavy data
  python -m src.pipeline.pipeline

  # 2. Tell DVC to track the directory and push to AWS
  dvc add artifacts/data_ingestion/raw/
  dvc push

  # 3. Commit only the pointer metadata to Git
  git add artifacts/data_ingestion/raw.dvc .gitignore
  git commit -m "Track raw dataset via DVC S3 remote"

```

Anyone cloning the repo gets instant access to the exact dataset snapshot corresponding to any Git commit using `dvc pull`.

---

## ⚙️ Pipeline Execution Stages

### Stage 01: Data Ingestion (`data_ingestion`)

* Validates local target directories.
* Checks if `creditcard.csv` is already cached locally; if not, fetches the dataset from remote source.
* Writes DVC metadata pointers to register raw data state.

### Stage 02: Data Preparation (`data_preparation`)

* **Stratified Split:** Enforces `StratifiedShuffleSplit` (80/20) to guarantee the 0.173% fraud proportion is precisely maintained in both training and test sets.
* **Feature Scaling:** Uses `RobustScaler` on `Amount` and cyclical feature engineering on `Time`.
* **Resampling (Optional / Parameterized):** Applies Synthetic Minority Over-sampling Technique (`SMOTE`) strictly to `train.parquet` to synthesize minority fraud instances in feature space without inflating `test.parquet`.
* **Serialization:** Saves structured partitions as high-performance, compressed columnar Parquet files.

### Stage 03: Model Training (`data_training`)

* Loads `train.parquet` and configures **XGBoost (`XGBClassifier`)**.
* Implements cost-sensitive weighting via `scale_pos_weight` to penalize false negatives heavily.
* Serializes the trained estimator to `artifacts/data_training/model.joblib`.
* Computes baseline validation scores to `metrics.json`.

---

## 📈 Model Performance & Results Analysis

Given the severe fraud imbalance, traditional metric interpretation is dangerous:

| Metric | Score | Analysis |
| --- | --- | --- |
| **ROC-AUC** | **~0.97 - 0.98** | High overall separability across thresholds, but can mask false positives due to the enormous True Negative pool. |
| **PR-AUC (Average Precision)** | **~0.82 - 0.86** | The gold standard for imbalanced fraud classification. Demonstrates that the model keeps precision high even at steep recall requirements. |
| **Recall (Fraud Class)** | **~0.85 - 0.90** | Caught 85–90% of all real fraudulent transactions without letting them slip past the system. |
| **Precision (Fraud Class)** | **~0.80 - 0.85** | Out of all transactions flagged as fraud, over 80% were true attacks, avoiding unacceptable customer friction. |

### Confusion Matrix Insights (Test Set Evaluation)

* **True Negatives (Legitimate Approved):** Over 56,800 clean transactions processed without incident.
* **False Positives (False Alarms):** Under 20 legitimate transactions flagged for secondary verification.
* **False Negatives (Missed Fraud):** Under 12 fraudulent events slipped past baseline detection—minimizing financial chargeback losses.

---

## 📂 Project Directory Structure

```text
churn-mlops-pipeline/
├── .dvc/                         # DVC configuration and remote cache references
│   └── config
├── artifacts/                    # Local binary store (Ignored by Git, tracked by DVC)
│   ├── data_ingestion/
│   ├── data_preparation/
│   └── data_training/
├── config/
│   └── config.yaml              # Pipeline stage paths & artifact destinations
├── params.yaml                   # Hyperparameters (learning_rate, estimators, scaling)
├── src/
│   └── pipeline/
│       ├── components/           # Core algorithmic stage classes
│       │   ├── data_ingestion.py
│       │   ├── data_preparation.py
│       │   └── data_training.py
│       ├── config/               # Configuration loading & schema verification
│       │   └── configuration_manager.py
│       ├── entity/               # Immutable Dataclass configs
│       │   └── config_entity.py
│       ├── logger.py             # Centralized structured application logging
│       └── pipeline.py           # End-to-end stage orchestration runner
├── .gitignore
├── requirements.txt
├── setup.py                      # Local editable package installer
└── README.md

```

---

## 🚀 Installation & Setup

### 1. Clone & Set Up Python Environment

```bash
git clone [https://github.com/arpitagrawal/churn-mlops-pipeline.git](https://github.com/arpitagrawal/churn-mlops-pipeline.git)
cd churn-mlops-pipeline

# Recommended: Python 3.9+ Conda or Virtualenv
conda create -n churn-mlops python=3.9 -y
conda activate churn-mlops

```

### 2. Install Package in Editable Mode

```bash
pip install -r requirements.txt
pip install -e .

```

### 3. Configure AWS S3 Remote Access (for DVC)

Ensure your AWS credentials are set up with permissions to access your S3 bucket:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="your-region"

# Pull versioned artifacts from S3
dvc pull

```

---

## 🔁 Reproducibility & Execution

Run the complete pipeline end-to-end from the project root:

```bash
python -m src.pipeline.pipeline

```

To adjust model hyperparameters or data splitting ratios, modify **`params.yaml`** directly without modifying Python code.

---

## 👤 Author

* **Arpit Agrawal** — MSc Data Science, TU Dortmund
* GitHub: [@arpit1507](https://www.google.com/search?q=https://github.com/arpit1507)

```