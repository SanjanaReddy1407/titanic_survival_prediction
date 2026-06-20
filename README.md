# Titanic: Machine Learning from Disaster 🚢

This repository contains a full-stack Machine Learning web application that predicts a passenger's survival chance on the Titanic based on their demographics and ticket characteristics. The project includes a data preprocessing pipeline, an optimized **Random Forest Classifier**, and an interactive web user interface built with **Flask (Python), HTML, CSS, and JavaScript**.

---

## 📊 Model Performance Metrics
The final trained Random Forest model achieves a high-performance baseline on unseen test data, indicating well-balanced generalizations:

* **Overall Accuracy:** `86.59%`
* **Precision:** `81.94%`
* **Recall:** `84.26%`
* **F1-Score:** `76.92%`

---

## 🛠️ Data Preprocessing & Feature Engineering
Before training the model, the raw historical dataset was thoroughly cleaned and transformed using `pandas`:
* **Missing Value Imputation:** Missing values in the `Age` feature were imputed using the floored mean age of the dataset to maintain consistency. Missing `Embarked` values were handled using the column's statistical mode.
* **Categorical Mapping:** Non-numeric categories were structurally mapped to numeric values (`Sex` ➔ `male: 0, female: 1`; `Embarked` ➔ `S: 1, C: 2, Q: 3`).
* **Feature Engineering (`Family` Size):** Extracted hidden signals from family relationships by combining sibling/spouse counters and parent/child counters into a consolidated feature:
  $$\text{Family} = \text{SibSp} + \text{Parch} + 1$$
* **Dimensionality Reduction:** Irrelevant features such as `Name`, `PassengerId`, `Ticket`, and `Cabin` were dropped to minimize noise and improve performance.

---

## 📂 Project Architecture
```text
TitanicApp/
│
├── Titanic-Dataset.csv    # Raw historical passenger records
├── app.py                 # Flask server backend & model training pipeline
└── templates/
    └── index.html         # Responsive frontend user interface with CSS animations
