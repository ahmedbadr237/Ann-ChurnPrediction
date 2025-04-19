# Customer Churn Prediction using ANN

This project is an end-to-end implementation of an Artificial Neural Network (ANN) model to predict customer churn in a banking dataset. The model is built using TensorFlow/Keras and includes data preprocessing techniques like robust scaling and outlier capping for improved performance and stability.

---

## 🔍 Problem Statement

Customer churn is a critical problem for businesses, especially in the banking sector. The goal is to predict whether a customer will exit (churn) the bank based on their demographic and account-related features.

---

## 📊 Dataset Overview

The dataset includes the following features:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (Target variable)

---

## ⚙️ Preprocessing Steps

To ensure data quality and enhance model performance, the following preprocessing techniques were applied:

- **Outlier Handling**:
  - Applied **capping** to limit extreme outlier values in numerical features.
- **Feature Scaling**:
  - Used **RobustScaler** from `sklearn.preprocessing` to scale features while minimizing the influence of outliers.
- **Categorical Encoding**:
  - Converted categorical variables (`Geography`, `Gender`) using **One-Hot Encoding** or **Label Encoding**, depending on the algorithm’s requirement.

---

## 🧠 Model Architecture (ANN)

The Artificial Neural Network was built using Keras and consists of:

- Input layer matching the number of input features
- Two hidden layers with ReLU activation
- Output layer with sigmoid activation for binary classification

The model was compiled with:

- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

---

## 📈 Performance

The model was evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

Results indicate a reliable model for predicting churn probability, helping businesses identify at-risk customers.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ann-churn-prediction.git
   cd ann-churn-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
  ```bash
  python app.py
  ```
