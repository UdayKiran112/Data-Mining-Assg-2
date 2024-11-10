import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X.values:
            # Calculate distances to all training points
            distances = []
            for x_train in self.X_train.values:
                dist = self.euclidean_distance(x, x_train)
                distances.append(dist)

            # Find k nearest neighbors
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train.iloc[k_indices]

            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return predictions


def evaluate_knn_scratch():
    # Load data
    train_data = pd.read_csv("../Data Preparation/train.csv")
    test_data = pd.read_csv("../Data Preparation/test.csv")

    # Separate features and target
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Train and evaluate model
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)

    # Predictions on test data
    y_test_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)

    print("KNN from Scratch - Test Results:")
    print(f"Confusion Matrix:\n{test_conf_matrix}")
    print(f"\nAccuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")

    # Predictions on training data for overfitting check
    y_train_pred = knn.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average="weighted")
    train_recall = recall_score(y_train, y_train_pred, average="weighted")

    print("\nKNN from Scratch - Training Results:")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")

    # Check for overfitting or underfitting
    if train_accuracy - test_accuracy > 0.1:
        print("\nModel may be overfitting.")
    elif test_accuracy - train_accuracy > 0.1:
        print("\nModel may be underfitting.")
    else:
        print("\nModel has balanced performance on training and test data.")


if __name__ == "__main__":
    evaluate_knn_scratch()
