# Watt-the-Predictor : Energy Consumption Prediction

This project predicts building energy consumption using the ASHRAE dataset. It explores multiple ML models and deploys the best ones using an interactive Streamlit web app.

---

## 📌 Project Overview

The goal is to build predictive models for energy usage (electricity, chilled water, steam, hot water) based on:
- 🏢 Building metadata
- 🌦️ Weather conditions
- 🕒 Timestamp features (day, hour, etc.)

The app lets you:
- Select different trained models
- View model evaluation metrics (RMSE, MAE, RMSLE, R²)
- See feature importances (for tree-based models)
- Input values manually or upload a CSV
- Get predictions + download results

---

## 🧰 Tech Stack

### ⚙️ Machine Learning & Data Processing
- **Pandas** – Data manipulation and preprocessing  
- **NumPy** – Numerical operations  
- **Scikit-learn** – Model building, evaluation, hyperparameter tuning  
- **Keras / TensorFlow** – Neural network modeling  
- **Joblib** – Model serialization and loading  

### 📊 Visualization
- **Matplotlib / Seaborn** – For offline plotting and analysis  
- **Streamlit** – For creating interactive web app dashboards  
- **Streamlit Charts** – Built-in bar charts for feature importance  

### 🧪 Model Evaluation
- RMSE, MAE, RMSLE, R² metrics  
- GridSearchCV for hyperparameter tuning

---

## 🧠 Models Used

| Model                     | Description |
|--------------------------|-------------|
| **Baseline (Median)**    | Predicts median of training data |
| **Linear Regression**    | Basic linear model |
| **Ridge Regression**     | Linear model with L2 regularization |
| **Lasso Regression**     | Linear model with L1 regularization |
| **ElasticNet**           | Combines L1 + L2 regularization |
| **SGD Regressor**        | Optimizes loss via stochastic gradient descent |
| **Decision Tree (Tuned)**| Tree-based model optimized with GridSearchCV |
| **Neural Network (Keras)**| Multi-layered perceptron with early stopping |

---

## 🚀 Web App Features

Built using **Streamlit**.  
Includes:

- ✅ Model selection via dropdown
- 📊 Card-style evaluation metric display
- 🔍 Feature importance chart (for trees)
- 📝 Manual input or 📁 CSV upload
- 📥 Download results as CSV

## 🛠️ How to Run the App

1. Clone the repo:
    ```bash
    git clone https://github.com/addittidas/watt-the-predictor.git
    cd watt-the-predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download dataset from Kaggle:
    ```bash
    https://www.kaggle.com/competitions/ashrae-energy-prediction/data
    ```

4. Run notebooks to train models and save in folder named models:
    ```bash
    jupyter notebooks
    ```

5. Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
---

## 📷 Web App Pic

![Main Page](main_page.png)
> For example: Home screen, predictions table, feature importance chart.

---

## 📄 License

This project is licensed under the MIT License.

---


