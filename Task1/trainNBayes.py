import pandas as pd
import numpy as np
import json


class NaiveBayesDiscretizer:
    def discretize(self, X, feature_names):
        self.bins = {}
        discretized_X = pd.DataFrame()

        for feature in feature_names:
            data = X[feature]

            # Check if the feature has variation
            if data.nunique() > 1:
                bins = np.linspace(data.min(), data.max(), num=4)  # 3 equal-width bins
                self.bins[feature] = bins
                discretized_X[feature] = np.digitize(data, bins[:-1])
            else:
                # If feature has no variation, put all values in the same bin
                self.bins[feature] = [data.min(), data.max()]
                discretized_X[feature] = 1  # Assign a single bin for all values

        return discretized_X

    def save_bins(self, filename):
        # Convert all ndarray bins to lists for JSON serialization
        bins_as_lists = {feature: bins.tolist() for feature, bins in self.bins.items()}
        with open(filename, "w") as f:
            json.dump(bins_as_lists, f)


def train_naive_bayes():
    # Load training data
    train_data = pd.read_csv("../Data Preparation/train.csv")

    # Separate features and target
    X = train_data.drop("target", axis=1)
    y = train_data["target"]

    # Discretize features
    discretizer = NaiveBayesDiscretizer()
    X_discrete = discretizer.discretize(X, X.columns)
    discretizer.save_bins("bins.txt")

    # Calculate prior probabilities with Laplace smoothing
    class_counts = y.value_counts()
    num_classes = len(class_counts)
    priors = (class_counts + 1) / (len(y) + num_classes)

    # Calculate likelihood probabilities with Laplace smoothing
    likelihoods = {}
    for feature in X_discrete.columns:
        likelihoods[feature] = {}
        for class_label in class_counts.index:  # Use class labels directly (strings)
            feature_counts = X_discrete.loc[y == class_label, feature].value_counts()
            # Add Laplace smoothing
            smoothed_probs = (feature_counts + 1) / (
                class_counts[class_label] + 3
            )  # 3 bins
            # Convert bin numbers to strings in the dictionary
            likelihoods[feature][class_label] = {
                str(int(k)): float(v) for k, v in smoothed_probs.to_dict().items()
            }

    # Save probabilities
    probabilities = {
        "priors": {str(k): float(v) for k, v in priors.to_dict().items()},
        "likelihoods": likelihoods,
    }

    with open("probabilities.txt", "w") as f:
        json.dump(probabilities, f)


if __name__ == "__main__":
    train_naive_bayes()
