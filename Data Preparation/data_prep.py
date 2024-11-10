import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset():
    try:
        # Load the iris data from the local file
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Read the data file
        df = pd.read_csv('./iris/iris.data', names=feature_names + ['target'], header=None)
        
        print("Data loaded successfully.")
        
        # Display feature information
        print("\nFeatures preview:")
        print(df[feature_names].head())
        
        # Display target information
        print("\nTargets preview:")
        print(df[['target']].head())
        
        print("\nCombined DataFrame preview:")
        print(df.head())
        
        print("\nFeature statistics:")
        print(df[feature_names].describe())
        
        print("\nTarget distribution:")
        print(df['target'].value_counts())
        
        # Split into train (75%) and test (25%) with stratification
        train_data, test_data = train_test_split(
            df, 
            test_size=0.25, 
            stratify=df['target'],
            random_state=42
        )
        
        # Further split train into train (70% of total) and validation (5% of total)
        train_final, valid_data = train_test_split(
            train_data,
            test_size=0.05/0.75,  # This ensures validation is 5% of total dataset
            stratify=train_data['target'],
            random_state=42
        )
        
        # Save datasets to CSV files
        train_final.to_csv('train.csv', index=False)
        valid_data.to_csv('train-valid.csv', index=False)
        test_data.to_csv('test.csv', index=False)
        
        print("\nData split sizes:")
        print(f"Total dataset size: {len(df)} samples")
        print(f"Training set: {len(train_final)} samples")
        print(f"Validation set: {len(valid_data)} samples")
        print(f"Test set: {len(test_data)} samples")
        
        # Verify target distribution in splits
        print("\nTarget distribution in splits:")
        print("Training set:")
        print(train_final['target'].value_counts())
        print("\nValidation set:")
        print(valid_data['target'].value_counts())
        print("\nTest set:")
        print(test_data['target'].value_counts())
        
        print("\nData preparation complete.")
        return train_final, valid_data, test_data
    
    except FileNotFoundError:
        print("Error: Could not find the iris.data file. Please ensure it's in the same directory as this script.")
        print("Available files should include one of: iris.data, bezdekIris.data")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

if __name__ == "__main__":
    prepare_dataset()