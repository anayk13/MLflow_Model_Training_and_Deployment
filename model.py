import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set up logging to display MLflow logs
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(_name_)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Ensure this matches your server

# Define the function to evaluate and plot accuracy
def plot_accuracy_graph(accuracies):
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title('Model Accuracy over Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.show()

if _name_ == "_main_":
    warnings.filterwarnings("ignore")

    # Load the wine dataset directly from the folder
    data = pd.read_csv("/Users/anaykumar/Desktop/MLflow/wine-quality.csv", sep=";")

    # Convert quality to binary classes for classification: Good (>=6) and Bad (<6)
    data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

    # Split the dataset
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop("quality", axis=1)
    train_y = train["quality"]
    test_x = test.drop("quality", axis=1)
    test_y = test["quality"]

    # Hyperparameters
    n_estimators = 50  # Adjust or use command-line arguments if needed
    max_depth = 10
    min_samples_split = 2

    # Start MLflow run
    with mlflow.start_run():
        # Initialize and train the model
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        
        accuracies = []
        for i in range(1, n_estimators + 1):
            rf.set_params(n_estimators=i)
            rf.fit(train_x, train_y)
            train_pred = rf.predict(train_x)
            accuracy = accuracy_score(train_y, train_pred)
            accuracies.append(accuracy)

        # Final model evaluation on test data
        test_pred = rf.predict(test_x)
        test_accuracy = accuracy_score(test_y, test_pred)
        print(f"Test Accuracy: {test_accuracy}")
        print(classification_report(test_y, test_pred))

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log accuracy graph
        plt.figure()
        plot_accuracy_graph(accuracies)
        accuracy_graph_path = "accuracy_graph.png"
        plt.savefig(accuracy_graph_path)
        mlflow.log_artifact(accuracy_graph_path)

        # Log model
        mlflow.sklearn.log_model(rf, "model", registered_model_name="RandomForestWineQualityClassifier")

        # Clean up
        os.remove(accuracy_graph_path)

        print("Model and artifacts logged successfully in MLflow.")