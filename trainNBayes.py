# trainNBayes.py

import pandas as pd
import numpy as np

# Load training data
train_data = pd.read_csv("train.csv")

# Discretize numerical features into 3 equal-width bins
def discretize_features(data, feature_columns):
    bins = 3
    for column in feature_columns:
        data[column] = pd.cut(data[column], bins=bins, labels=False)
    return data

# Calculate prior probabilities and likelihoods with Laplacian correction
def calculate_probabilities(train_data, feature_columns, target_column='species'):
    priors = train_data[target_column].value_counts(normalize=True).to_dict()
    likelihoods = {}

    # Calculate likelihoods for each species and each feature bin
    for species in train_data[target_column].unique():
        species_data = train_data[train_data[target_column] == species]
        likelihoods[species] = {}
        for feature in feature_columns:
            # Get bin counts with Laplacian correction
            bin_counts = species_data[feature].value_counts()
            total_count = len(species_data) + len(bin_counts)  # Laplacian correction
            likelihoods[species][feature] = ((bin_counts + 1) / total_count).to_dict()
    
    return priors, likelihoods

# Discretize features
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
train_data = discretize_features(train_data, feature_columns)

# Calculate probabilities
priors, likelihoods = calculate_probabilities(train_data, feature_columns)

# Save probabilities to a text file
with open("probabilities.txt", "w") as file:
    file.write("Priors:\n")
    for species, prob in priors.items():
        file.write(f"{species}: {prob}\n")
    file.write("\nLikelihoods:\n")
    for species, features in likelihoods.items():
        file.write(f"{species}:\n")
        for feature, probs in features.items():
            file.write(f"  {feature}: {probs}\n")

print("Priors and likelihoods saved to probabilities.txt.")
