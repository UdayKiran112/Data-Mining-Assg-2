import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def evaluate_knn_sklearn():
    # Load data
    train_data = pd.read_csv('train.csv')
    valid_data = pd.read_csv('train-valid.csv')
    test_data = pd.read_csv('test.csv')
    
    # Separate features and target
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_valid = valid_data.drop('target', axis=1)
    y_valid = valid_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
    }
    
    # Create and train model with grid search
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("\nTest Set Results:")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Analyze misclassified samples
    misclassified = test_data[y_test != y_pred]
    print("\nMisclassified samples:")
    print(misclassified)

if __name__ == "__main__":
    evaluate_knn_sklearn()
