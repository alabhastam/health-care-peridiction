# 🏥 Healthcare Cost Prediction with Neural Network (Regression)

This project focuses on predicting healthcare expenses based on personal and lifestyle attributes such as age, BMI, smoking habits, etc. The dataset is used to train a regression model that generalizes well to unseen data.

---

## 📊 Dataset Description

The dataset contains the following columns:

- `age`: Age of the individual
- `sex`: Gender (`male`, `female`)
- `bmi`: Body Mass Index
- `children`: Number of children
- `smoker`: Whether the person smokes (`yes`, `no`)
- `region`: Geographic region (`northeast`, `southeast`, `southwest`, `northwest`)
- `expenses`: Annual medical expenses (target variable)

---

## 🧪 Goal

> Predict the `expenses` column using the other features.

The model should achieve a **Mean Absolute Error (MAE) under $3500** on unseen data.

---

## 📦 Libraries Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn (for visualization)

---

## 🔧 Preprocessing Steps

1. **Load the data**
2. **Convert categorical variables** (`sex`, `smoker`, `region`) using one-hot encoding (`pd.get_dummies`)
3. **Split** the data into:
   - `80%` training data
   - `20%` testing data
4. **Separate the labels** (`expenses`) from features
5. **Normalize** the feature values using `StandardScaler`

---

## 🤖 Model: Neural Network (Keras)

A simple feed-forward neural network with:

- `Dense(64)` → ReLU
- `Dense(64)` → ReLU
- `Dense(1)` → Linear (regression output)

