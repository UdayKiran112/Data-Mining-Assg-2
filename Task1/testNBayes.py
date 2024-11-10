import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

def load_probabilities():
    with open('probabilities.txt', 'r') as f:
        return json.load(f)

def load_bins():
    with open('bins.txt', 'r') as f:
        return json.load(f)

def discretize_test_data(X, bins):
    discretized_X = pd.DataFrame()
    for feature in X.columns:
        discretized_X[feature] = np.digitize(X[feature], np.array(bins[feature])[:-1])
    return discretized_X

def predict(X_discrete, probabilities):
    predictions = []
    for _, sample in X_discrete.iterrows():
        class_scores = {}
        for class_label in probabilities['priors'].keys():
            score = np.log(probabilities['priors'][class_label])
            for feature in X_discrete.columns:
                bin_value = str(int(sample[feature]))  # Convert bin value to string
                likelihood = probabilities['likelihoods'][feature][class_label].get(
                    bin_value, 1/(len(probabilities['priors']) + 3)  # Laplace smoothing
                )
                score += np.log(likelihood)
            class_scores[int(class_label)] = score
        predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])
    return predictions

def evaluate_model():
    # Load test data
    test_data = pd.read_csv('test.csv')
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Load probabilities and bins
    probabilities = load_probabilities()
    bins = load_bins()
    
    # Discretize test data
    X_test_discrete = discretize_test_data(X_test, bins)
    
    # Make predictions
    y_pred = predict(X_test_discrete, probabilities)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Analyze misclassified samples
    misclassified = test_data[y_test != y_pred]
    print("\nMisclassified samples:")
    print(misclassified)

if __name__ == "__main__":
    evaluate_model()