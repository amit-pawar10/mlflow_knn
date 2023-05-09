import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import datetime
import os

import matplotlib.pyplot as plt

def generate_plot(X_test, y_test, model, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # Generate predictions
    y_pred = model.predict(X_test)

    # Create a scatter plot of true values vs. predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True Values vs. Predicted Values')

    # Save the plot to the specified plot path
    plt.savefig(plot_path)
    plt.close()


def train_model(data_path, model_path):
    # Load the Boston Housing dataset
    dataset = fetch_openml(name='boston')
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Generate a unique model name using the current timestamp
    model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Convert X_test to NumPy array
    X_test = np.array(X_test)

    # Calculate metrics on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model", "linear_regression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Generate a plot and save it as an artifact
    plot_path = "path/to/your/plot.png"
    generate_plot(X_test, y_test, model, plot_path)
    mlflow.log_artifact(plot_path)

    # Save the model
    mlflow.sklearn.save_model(model,  f"{model_path}/{model_name}")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/boston_house_prices.csv")
    parser.add_argument("--model-path", type=str, default="models")
    args = parser.parse_args()

    # Start an MLflow run
    with mlflow.start_run():
        # Train the model and log parameters/metrics
        train_model(args.data_path, args.model_path)
