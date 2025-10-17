import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import fire

def evaluate_classification(csv_file_path):
    """
    Evaluate classification performance metrics
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file_path, header=0)
    
    # Extract true labels (second-to-last column) and predicted labels (last column)
    y_true = df.iloc[:, -2].values  # Second-to-last column
    y_pred = df.iloc[:, -1].values  # Last column
    
    # Remove possible NaN values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Calculate various metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-class cases, use macro averaging
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Also calculate weighted averaging (considering class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Build result dictionary
    results = {
        'Accuracy': accuracy,
        'Precision (Macro)': precision,
        'Recall (Macro)': recall,
        'F1-Score (Macro)': f1,
        'Precision (Weighted)': precision_weighted,
        'Recall (Weighted)': recall_weighted,
        'F1-Score (Weighted)': f1_weighted,
        'Total Samples': len(y_true),
        'Number of Classes': len(np.unique(y_true))
    }
    
    return results

def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way
    """
    print("=" * 50)
    print("Classification Model Evaluation Results")
    print("=" * 50)
    
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.8f}")
        else:
            print(f"{metric}: {value}")
    
    print("=" * 50)

def detailed_evaluation(csv_file_path):
    """
    Detailed classification evaluation including confusion matrix and classification report
    """
    # Read data
    df = pd.read_csv(csv_file_path, header=None)
    y_true = df.iloc[:, -2].values  # Second-to-last column
    y_pred = df.iloc[:, -1].values  # Last column
    
    # Remove NaN values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Basic evaluation
    results = evaluate_classification(csv_file_path)
    print_evaluation_results(results)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return results

def main(csv_path):
    results = evaluate_classification(csv_path)
    print_evaluation_results(results)


# Usage example
if __name__ == "__main__":
    fire.Fire(main)
  