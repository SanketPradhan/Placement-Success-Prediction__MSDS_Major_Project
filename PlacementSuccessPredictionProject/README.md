# 🚀 Placement Success Prediction

![Placement Success Prediction Banner](assets/readme/banner.svg)

> An interpretable machine learning system for forecasting placement outcomes with a strong emphasis on transparency, usability, and real-world deployment readiness.

This repository presents an end-to-end solution for predicting whether a student is likely to achieve a strong placement outcome. The project combines data preprocessing, model training, evaluation, explainability, and a Flask-based user interface into a cohesive product-style workflow suitable for both academic study and portfolio presentation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FFB300?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-68.15%25-brightgreen?style=for-the-badge)
![Deployment](https://img.shields.io/badge/Deployment-Flask%20App-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## ✨ Why this project matters

This initiative is designed to bridge the gap between predictive analytics and decision support. By combining robust model performance with explainability, it helps stakeholders understand not only what the model predicts, but also why it makes those predictions.

- Predicts placement outcomes using academic, technical, and professional profile features
- Trains a high-performing XGBoost classification pipeline
- Evaluates model performance through confusion matrices and ROC analysis
- Offers an interactive Flask interface for real-time inference
- Produces explainable AI artifacts for transparency and stakeholder trust

---

## 🧠 Product workflow

The system follows a structured, production-style workflow:

1. Ingest the student placement dataset
2. Clean and standardize features for model compatibility
3. Train an XGBoost classifier on the curated feature space
4. Evaluate predictive quality and save diagnostic visualizations
5. Package the trained pipeline for deployment through Flask
6. Generate explainability insights to support human-in-the-loop decision making

---

## 🗂️ Repository structure

```text
.
├── data/
│   └── student_placement_career_success_dataset_2026.csv
├── models/
│   ├── placement_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── xgb_model.pkl
│   └── disease_model.pkl
├── notebooks/
│   └── PlacementSuccessPrediction.ipynb
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   ├── feature_importance.csv
│   └── lime_explanation.html
├── src/
│   ├── app.py
│   ├── data_ingestion.py
│   ├── evaluate_models.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── generate_explainability.py
│   ├── preprocessing.py
│   └── train_models.py
├── templates/
│   └── index.html
└── README.md
```

---

## 🛠️ Technical stack

- Python
- Flask
- pandas
- scikit-learn
- XGBoost
- matplotlib
- joblib
- SHAP
- LIME

---

## ⚡ Quick start

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd PlacementSuccessPredictionProject
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install flask pandas scikit-learn xgboost matplotlib joblib lime shap
```

---

## ▶️ Run the web app

```bash
python src/app.py
```

Then open your browser at:

```text
http://127.0.0.1:5000
```

The app allows users to enter student profile details and receive:

- predicted placement outcome
- success probability
- failure probability
- a decision rule explanation

---

## 🧪 Train the model

To retrain the model and regenerate evaluation artifacts:

```bash
python src/train_models.py
```

This will:

- train the XGBoost pipeline
- save the trained model to the models folder
- generate performance charts in the reports folder
- export feature importance results

---

## 📊 Model performance

The current trained model achieves an accuracy of 68.15% on the held-out evaluation set, with balanced classification performance across both outcome classes.

### Evaluation highlights

- Accuracy: 68.15%
- Deployment status: Flask-based inference interface available
- Model type: XGBoost classifier with preprocessing pipeline
- Explainability: feature importance, LIME, and SHAP-ready outputs included

## 🖼️ Demo screenshot

A representative view of the prediction interface is shown below:

![Demo Screenshot](assets/readme/confusion_matrix.png)

> The interface is designed to make predictions accessible while preserving clarity around the model’s output.

## 📊 Model details

The project uses a machine learning pipeline with:

- numerical feature scaling
- categorical encoding
- XGBoost classification

Key input features include:

- Age
- CGPA
- DSA problems solved
- internships
- certifications
- projects count
- communication skills
- aptitude score
- coding platform ratings
- GitHub contributions
- hackathon participation
- AI/ML skill level
- resume and mock interview scores

The target can be derived from the placement status column or from a salary-based success threshold depending on the dataset configuration.

---

## 🔍 Explainability outputs

The project generates rich explainability artifacts in the reports directory:

- Feature importance CSV
- LIME explanation HTML
- SHAP summary plot
- ROC curve and confusion matrix images

These outputs help interpret which factors influence the prediction most.

---

## 📁 Generated artifacts

After training, you will find:

- models/placement_model.pkl — trained pipeline artifact
- reports/confusion_matrix.png — classification performance visual
- reports/roc_curve.png — ROC performance curve
- reports/feature_importance.csv — ranked important features
- reports/lime_explanation.html — local explanation example
- reports/shap_summary.png — global interpretability summary

---

## 🌟 Why this repository stands out

This project is well-suited for a professional portfolio because it demonstrates:

- end-to-end machine learning workflow design
- deployment-ready application architecture
- explainability-driven model interpretation
- product thinking from data preparation to user-facing prediction
- a strong balance of technical rigor and business relevance

---

## 🤝 Contributing

Contributions are welcome. If you would like to improve the model, expand the explainability layer, or enhance the interface, feel free to open a pull request or propose a feature direction.

---

## 📌 Notes

- The dataset used for training is located in [data/student_placement_career_success_dataset_2026.csv](data/student_placement_career_success_dataset_2026.csv)
- The app is intentionally simple and extensible for future enhancements such as authentication, dashboard views, or model monitoring

---

Built at the intersection of data science, machine learning, and applied AI product development.
