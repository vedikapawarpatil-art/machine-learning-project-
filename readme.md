# 🧠 Machine Learning Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Data-013243?logo=numpy&logoColor=white)](https://numpy.org/)

An end-to-end Machine Learning classification pipeline built with **Python** and **Scikit-Learn**. Demonstrates feature preprocessing, standard scaling, stratified data splitting, ensemble classification with Random Forest, rigorous evaluation, and model artifact serialization.

---

## 📌 Pipeline Overview

The pipeline executes five core stages:
1. **Dataset Ingestion**: Loads multidimensional diagnostic data with 30 quantitative features.
2. **Stratified Splitting**: 80/20 train-test partition maintaining class distribution.
3. **Feature Normalization**: Standardizes feature variances using `StandardScaler`.
4. **Model Architecture**: Trains an ensemble `RandomForestClassifier` with balanced depth to prevent overfitting.
5. **Evaluation & Persistence**: Generates precision, recall, F1-scores, and confusion matrices before serializing model artifacts with `joblib`.

---

## 📊 Evaluation Metrics

When trained on the benchmark Wisconsin diagnostic dataset, the model achieves:
- **Test Accuracy**: **> 96%**
- **Precision (Malignant / Benign)**: Balanced across both classes
- **Artifacts Generated**: `models/random_forest_model.joblib`, `models/scaler.joblib`

---

## 🚀 Quickstart

### 1. Clone & Set Up
```bash
git clone https://github.com/vedikapawarpatil-art/machine-learning-project-.git
cd machine-learning-project-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
python train.py
```

---

## 📂 Project Structure

```text
machine-learning-project-/
├── models/                     # Serialized model & scaler binaries
├── requirements.txt            # Project dependencies
├── train.py                    # Pipeline execution script
└── readme.md                   # Project documentation
```

---

## 👤 Author

**Vedika Pawar-Patil**
- GitHub: [@vedikapawarpatil-art](https://github.com/vedikapawarpatil-art)
- LinkedIn: [Vedika Pawar-Patil](https://www.linkedin.com/in/vedika-pawar-patil-8681a5338)
