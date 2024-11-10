import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


def evaluate_knn_sklearn():
    # Load data
    train_data = pd.read_csv("../Data Preparation/train.csv")
    valid_data = pd.read_csv("../Data Preparation/train-valid.csv")
    test_data = pd.read_csv("../Data Preparation/test.csv")

    # Separate features and target
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_valid = valid_data.drop("target", axis=1)
    y_valid = valid_data["target"]
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],  # p=1 for manhattan, p=2 for euclidean
    }

    # Create and train model with grid search
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train_scaled, y_train)

    # Print best parameters
    print("Best parameters:", grid_search.best_params_)

    # Evaluate on training, validation, and test sets
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train_scaled)
    y_valid_pred = best_model.predict(X_valid_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    # Calculate metrics for training, validation, and test sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    print("\nTest Set Results:")
    print(f"Confusion Matrix:\n{test_conf_matrix}")
    print(f"\nAccuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    
    # Display detailed classification report (including per-class precision, recall, f1-score)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # Display train and validation accuracy for overfitting/underfitting check
    print("\nPerformance Summary:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Overfitting/Underfitting Analysis
    if train_accuracy - valid_accuracy > 0.05:  # Adjust the threshold as needed
        print("\nThe model may be overfitting, as training accuracy is significantly higher than validation accuracy.")
    elif valid_accuracy > train_accuracy:
        print("\nThe model may be underfitting, as validation accuracy is higher than training accuracy.")
    else:
        print("\nThe model shows balanced performance across training and validation sets.")

    # Analyze misclassified samples
    misclassified = test_data[y_test != y_test_pred]
    print("\nMisclassified samples:")
    print(misclassified)


if __name__ == "__main__":
    evaluate_knn_sklearn()
