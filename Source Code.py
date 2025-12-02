import numpy as np
from sklearn import datasets, linear_model, metrics, model_selection
import matplotlib.pyplot as plt
import os

def load_and_split_data():
    """
    Loads the diabetes dataset and splits it into training and testing sets.
    """
    X, y = datasets.load_diabetes(return_X_y=True)
    # Split 80/20 as per the original notebook
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Trains a standard OLS Linear Regression model."""
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_ridge_cv(X_train, y_train, alphas):
    """
    Trains a Ridge Regression model with built-in Cross-Validation 
    to find the best alpha.
    """
    # RidgeCV performs Leave-One-Out Cross-Validation by default
    ridge_cv = linear_model.RidgeCV(alphas=alphas)
    ridge_cv.fit(X_train, y_train)
    print(f"  > Best Alpha for Ridge: {ridge_cv.alpha_:.4f}")
    return ridge_cv

def train_lasso_cv(X_train, y_train, alphas):
    """
    Trains a Lasso Regression model with built-in Cross-Validation
    to find the best alpha.
    """
    # LassoCV uses K-Fold CV (default 5-fold)
    lasso_cv = linear_model.LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_train, y_train)
    print(f"  > Best Alpha for Lasso: {lasso_cv.alpha_:.4f}")
    return lasso_cv

def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Calculates MSE and R2 score for a given model.
    """
    y_pred = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    
    print(f"--- {name} Performance ---")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    return mse, r2

def plot_coefficients(model, feature_names, filename):
    """
    Plots the coefficients of the model to show feature importance.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, model.coef_)
    plt.title(f'{filename} Coefficients')
    plt.ylabel('Coefficient Value')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/{filename}_coeffs.png')
    plt.close()

def main():
    print("Starting Diabetes Prediction Pipeline...")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_split_data()
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    
    # Define range of alphas for regularization (logarithmic scale)
    alphas = np.logspace(-4, 4, 100)

    # 2. Train Models
    print("\nTraining Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    
    print("\nTraining Ridge Regression (with CV)...")
    ridge_model = train_ridge_cv(X_train, y_train, alphas)
    
    print("\nTraining Lasso Regression (with CV)...")
    lasso_model = train_lasso_cv(X_train, y_train, alphas)

    # 3. Evaluate Models
    print("\nEvaluation Results:")
    evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    evaluate_model(ridge_model, X_test, y_test, "Ridge CV")
    evaluate_model(lasso_model, X_test, y_test, "Lasso CV")

    # 4. Save Coefficient Plots
    print("\nGenerating coefficient plots in /results folder...")
    plot_coefficients(ridge_model, feature_names, "Ridge")
    plot_coefficients(lasso_model, feature_names, "Lasso")
    
    print("\nPipeline Complete.")

if __name__ == "__main__":
    main()