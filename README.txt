Diabetes Progression Prediction

📌 Project Overview

This project focuses on predicting the disease progression of diabetes patients based on ten baseline physiological variables. Using the Scikit-Learn Diabetes dataset, we implement and compare Ordinary Least Squares (OLS), Ridge Regression, and Lasso Regression to identify the most predictive features and control for overfitting.

🎯 Objectives

Predict a quantitative measure of disease progression one year after baseline.

Compare the performance of linear vs. regularized models (L2 Ridge, L1 Lasso).

Identify key physiological biomarkers using Lasso's feature selection capabilities.

🛠 Technologies

Python 3.9+

Scikit-Learn: Model implementation (LinearRegression, RidgeCV, LassoCV).

NumPy: Matrix operations.

Matplotlib: Visualization of regularization paths.

📂 Repository Structure

.
├── notebooks/          # Exploratory analysis and prototyping
│   └── exploration.ipynb  (Original academic notebook)
├── src/                # Production-ready source code
│   ├── train.py        # Main script for training and evaluation
│   └── utils.py        # Helper functions for plotting
├── results/            # Generated plots and metrics
├── requirements.txt    # Dependencies
└── README.md           # Project documentation


📊 Key Results

The models were evaluated using Mean Squared Error (MSE) and R² score on a held-out test set (20%).

Model

Test MSE

R² Score

Linear Regression

[Run script to generate]

[Run script to generate]

Ridge (Best $\alpha$)

[Run script to generate]

[Run script to generate]

Lasso (Best $\alpha$)

[Run script to generate]

[Run script to generate]

Key Insight: The Lasso model successfully performed feature selection, shrinking the coefficients of less relevant variables to zero, highlighting BMI and S5 (Lamotrigine) as strong predictors.

🚀 How to Run

Clone the repository:

git clone [https://github.com/yourusername/diabetes-prediction.git](https://github.com/yourusername/diabetes-prediction.git)
cd diabetes-prediction


Install dependencies:

pip install -r requirements.txt


Run the training pipeline:

python src/train.py
