import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)


def load_probabilities(file_path="probabilities.txt"):
    """Load probabilities for Naive Bayes from a text file."""
    with open(file_path, "r") as f:
        return eval(f.read())  # Using eval to handle non-JSON format


def load_bins(file_path="bins.txt"):
    """Load bin boundaries for discretization from a text file."""
    with open(file_path, "r") as f:
        return eval(f.read())  # Using eval to handle non-JSON format


def discretize_test_data(X, bins):
    """
    Discretize continuous features into bins based on pre-calculated boundaries.
    """
    discretized_X = pd.DataFrame()
    for feature in X.columns:
        # Digitize the feature values based on provided bins
        discretized_X[feature] = np.digitize(X[feature], np.array(bins[feature])[:-1])
    return discretized_X


def predict(X_discrete, probabilities):
    """
    Predict class labels for the given discretized data based on stored probabilities.
    """
    predictions = []
    for _, sample in X_discrete.iterrows():
        class_scores = {}
        for class_label, prior in probabilities["priors"].items():
            score = np.log(prior)
            for feature in X_discrete.columns:
                bin_value = str(
                    int(sample[feature])
                )  # Convert bin value to string for lookup
                # Retrieve likelihood with Laplace smoothing as a fallback
                likelihood = probabilities["likelihoods"][feature][class_label].get(
                    bin_value,
                    1 / (len(probabilities["priors"]) + probabilities["n_bins"]),
                )
                score += np.log(likelihood)
            # Use class_label directly without converting to int
            class_scores[class_label] = score
        # Select the class with the highest score
        predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])
    return predictions


def evaluate_model(test_file="../Data Preparation/test.csv"):
    """
    Evaluate the Naive Bayes classifier on test data and print performance metrics.
    """
    # Load test data and separate features and target
    test_data = pd.read_csv(test_file)
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Load saved model parameters
    probabilities = load_probabilities()
    bins = load_bins()

    # Discretize test data based on bins
    X_test_discrete = discretize_test_data(X_test, bins)

    # Make predictions
    y_pred = predict(X_test_discrete, probabilities)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Display misclassified samples
    misclassified = test_data[y_test != y_pred]
    if not misclassified.empty:
        print("\nMisclassified samples:")
        print(misclassified)
    else:
        print("\nNo misclassified samples.")


if __name__ == "__main__":
    evaluate_model()
