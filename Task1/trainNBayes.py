import pandas as pd
import numpy as np
import json


class NaiveBayesDiscretizer:
    def __init__(self, n_bins=3):
        self.bins = {}
        self.n_bins = n_bins

    def discretize(self, X, feature_names):
        """
        Discretize continuous features into bins using equal-width binning.
        """
        discretized_X = pd.DataFrame()

        for feature in feature_names:
            data = X[feature]

            # Check if the feature has variation
            if data.nunique() > 1:
                bins = np.linspace(data.min(), data.max(), num=self.n_bins + 1)
                self.bins[feature] = bins
                discretized_X[feature] = np.digitize(data, bins[:-1])
            else:
                self.bins[feature] = [data.min(), data.max()]
                discretized_X[feature] = 1

        return discretized_X

    def save_bins(self, filename):
        """Save bin boundaries to a text file."""
        bins_as_lists = {feature: bins.tolist() for feature, bins in self.bins.items()}
        with open(filename, "w") as f:
            f.write(str(bins_as_lists))


def train_naive_bayes(train_file="../Data Preparation/train.csv", n_bins=3):
    """
    Train Naive Bayes classifier with discretized features.
    """
    # Load training data
    train_data = pd.read_csv(train_file)

    # Separate features and target
    X = train_data.drop("target", axis=1)
    y = train_data["target"]

    # Initialize and apply discretizer
    discretizer = NaiveBayesDiscretizer(n_bins=n_bins)
    X_discrete = discretizer.discretize(X, X.columns)

    # Save discretization bins
    discretizer.save_bins("bins.txt")

    # Calculate prior probabilities with Laplace smoothing
    class_counts = y.value_counts()
    num_classes = len(class_counts)
    priors = (class_counts + 1) / (len(y) + num_classes)

    # Calculate likelihood probabilities with Laplace smoothing
    likelihoods = {}
    for feature in X_discrete.columns:
        likelihoods[feature] = {}
        for class_label in class_counts.index:
            feature_counts = X_discrete.loc[y == class_label, feature].value_counts()
            smoothed_probs = (feature_counts + 1) / (class_counts[class_label] + n_bins)
            likelihoods[feature][class_label] = {
                str(int(k)): float(v) for k, v in smoothed_probs.to_dict().items()
            }

    # Prepare probabilities dictionary
    probabilities = {
        "priors": {str(k): float(v) for k, v in priors.to_dict().items()},
        "likelihoods": likelihoods,
        "n_bins": n_bins,
    }

    # Save model parameters to text file
    with open("probabilities.txt", "w") as f:
        f.write(str(probabilities))

    print("Training completed successfully!")
    print(f"\nPrior probabilities:")
    for class_label, prior in probabilities["priors"].items():
        print(f"{class_label}: {prior:.4f}")

    print(f"\nFeature discretization bins:")
    for feature, bins in discretizer.bins.items():
        print(f"{feature}: {bins}")

    return discretizer, probabilities


if __name__ == "__main__":
    train_naive_bayes()
