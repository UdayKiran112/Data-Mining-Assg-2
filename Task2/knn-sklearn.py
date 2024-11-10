import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def load_probabilities(file_path):
    """ Load the probabilities from a txt file """
    probabilities = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Load the prior and likelihood probabilities
    for line in lines:
        class_label, *prob_values = line.strip().split(',')
        probabilities[class_label] = {
            'prior': float(prob_values[0]),
            'likelihood': {i: [float(prob_values[j*3 + 1]) for j in range(3)] for i in range(len(prob_values) // 3 - 1)}
        }
        
    return probabilities

def discretize_data(X):
    """ Discretize continuous features into 3 equal-width bins """
    binned_data = []
    for feature in X:
        min_val, max_val = feature.min(), feature.max()
        bin_width = (max_val - min_val) / 3
        binned_feature = np.digitize(feature, bins=[min_val + bin_width, min_val + 2*bin_width, max_val])
        binned_data.append(binned_feature)
    return np.array(binned_data).T

def predict(X, probabilities):
    """ Predict class labels using Naive Bayes """
    predictions = []
    
    for row in X:
        class_probabilities = {}
        
        for class_label, probs in probabilities.items():
            prior = probs['prior']
            likelihood = probs['likelihood']
            
            likelihoods = 1
            for idx, feature_value in enumerate(row):
                likelihoods *= likelihood.get(idx, [])[feature_value - 1]  # Get the likelihood for the feature's bin
            
            class_probabilities[class_label] = prior * likelihoods
        
        # Choose the class with the maximum probability
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        predictions.append(predicted_class)
    
    return predictions

def evaluate_model(y_true, y_pred):
    """ Evaluate the model by calculating accuracy, precision, recall, and confusion matrix """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\nTest Set Results:")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def main():
    # Load test data
    test_data = pd.read_csv('../Data Preparation/test.csv')
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Load pre-calculated probabilities
    probabilities = load_probabilities('probabilities.txt')
    
    # Discretize the test data
    X_test_discretized = discretize_data(X_test)
    
    # Predict class labels for the test set
    y_pred = predict(X_test_discretized, probabilities)
    
    # Evaluate model performance
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
