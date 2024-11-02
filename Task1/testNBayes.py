# testNBayes.py

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Updated load_probabilities function in testNBayes.py

def load_probabilities(filename="probabilities.txt"):
    priors = {}
    likelihoods = defaultdict(dict)
    with open(filename, "r") as file:
        lines = file.readlines()
        reading_priors = True
        for line in lines:
            line = line.strip()
            if not line or line == "Likelihoods:":
                reading_priors = False
                continue
            if reading_priors:
                if ": " in line:  # Check if the line has the expected format
                    species, prob = line.split(": ")
                    priors[species] = float(prob)
                else:
                    print(f"Skipping unexpected line in priors: {line}")
            else:
                if ":" in line:
                    if line.count(":") == 1:
                        # New species section
                        current_species = line[:-1]
                    else:
                        # Feature and probability mapping
                        feature, probs = line.split(": ", 1)
                        try:
                            likelihoods[current_species][feature] = eval(probs)
                        except:
                            print(f"Skipping unexpected line in likelihoods: {line}")
    return priors, likelihoods

# Load and discretize test data
def discretize_features(data, feature_columns):
    bins = 3
    for column in feature_columns:
        data[column] = pd.cut(data[column], bins=bins, labels=False)
    return data

# Naive Bayes prediction using loaded probabilities
def predict(priors, likelihoods, row, feature_columns):
    species_scores = {}
    for species, prior in priors.items():
        species_scores[species] = np.log(prior)  # Start with log prior
        for feature in feature_columns:
            feature_value = row[feature]
            feature_probs = likelihoods[species].get(feature, {})
            species_scores[species] += np.log(feature_probs.get(feature_value, 1e-6))  # Small value for unseen features
    return max(species_scores, key=species_scores.get)

# Main function to predict and evaluate on test set
def evaluate(test_data, priors, likelihoods):
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    test_data = discretize_features(test_data, feature_columns)
    
    # Predict classes
    test_data['predicted'] = test_data.apply(lambda row: predict(priors, likelihoods, row, feature_columns), axis=1)
    
    # Evaluation metrics
    y_true = test_data['species']
    y_pred = test_data['predicted']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:\n", conf_matrix)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

# Load test data
test_data = pd.read_csv("test.csv")

# Load priors and likelihoods
priors, likelihoods = load_probabilities()

# Evaluate model
evaluate(test_data, priors, likelihoods)
