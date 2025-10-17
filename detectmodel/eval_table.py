import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import fire
import os
import sys

def evaluate_classification(csv_file_path):
    """
    Evaluate classification performance metrics
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Warning: File not found: {csv_file_path}")
        return None
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path, header=0)
        
        # Extract true labels (second-to-last column) and predicted labels (last column)
        y_true = df.iloc[:, -2].values  # Second-to-last column
        y_pred = df.iloc[:, -1].values  # Last column
        
        # Remove possible NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            print(f"Warning: No valid data in {csv_file_path}")
            return None
        
        # Calculate accuracy
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
    
    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")
        return None

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

def evaluate_all_models(database_name, llm):
    """
    Evaluate all models and display results in table format
    
    Parameters:
    database_name (str): Name of the database
    llm (str): Name of the LLM model
    """
    
    attacks = ["badnets", "addsent", "synbkd", "stylebkd"]
    sample_sizes = ['none', '200', '400', '600', '800']
    test_types = ["test-clean", "test-poison"]

    if llm == "qwen":
        llm = "qwen_7b"
    elif llm == "llama3":
        llm = "llama3_8b"
    
    # Initialize results dictionary
    results = {}
    
    # Collect all results
    for attack in attacks:
        results[attack] = {}
        for sample_size in sample_sizes:
            results[attack][sample_size] = {}
            for test_type in test_types:
                if database_name == "sst-2":
                    csv_path = f"xxx/NLPLab/AgentsBD/bddata/{llm}_infer_de_{sample_size}/{attack}/{test_type}.csv"
                else:
                    csv_path = f"xxx/NLPLab/AgentsBD/bddata/{llm}_infer_de_{sample_size}_{database_name}/{attack}/{test_type}.csv"
                result = evaluate_classification(csv_path)
                if result:
                    results[attack][sample_size][test_type] = result['Accuracy']
                else:
                    results[attack][sample_size][test_type] = 0.0
    
    # Print table header
    print(f"\nEvaluation Results for {llm} on {database_name} database")
    print("=" * 120)
    
    # Create header
    header = f"{f'{database_name}/{llm}':<12}|"
    for attack in attacks:
        header += f"{attack:>20}|"
    print(header)
    
    # Create subheader
    subheader = f"{'':<12}|"
    for attack in attacks:
        subheader += f"{'clean':>9} {'attack':>10}|"
    print(subheader)
    
    print("-" * 120)
    
    # Print results for each sample size
    for sample_size in sample_sizes:
        row = f"{sample_size} samples|"
        for attack in attacks:
            clean_acc = results[attack][sample_size].get("test-clean", 0.0)
            poison_acc = results[attack][sample_size].get("test-poison", 0.0)
            row += f"{clean_acc:>9.8f} {poison_acc:>10.8f}|"
        print(row)
    
    print("=" * 120)
    
    # Also save results to CSV for further analysis
    # save_results_to_csv(results, llm, database_name)

def save_results_to_csv(results, llm, database_name):
    """
    Save results to CSV file for further analysis
    """
    rows = []
    attacks = ["badnets", "addsent", "synbkd", "stylebkd"]
    sample_sizes = [200, 400, 600, 800]
    
    for sample_size in sample_sizes:
        row = [f"{sample_size} samples"]
        for attack in attacks:
            clean_acc = results[attack][sample_size].get("test-clean", 0.0)
            poison_acc = results[attack][sample_size].get("test-poison", 0.0)
            row.extend([clean_acc, poison_acc])
        rows.append(row)
    
    # Create column names
    columns = ["Sample Size"]
    for attack in attacks:
        columns.extend([f"{attack}_clean", f"{attack}_attack"])
    
    df = pd.DataFrame(rows, columns=columns)
    output_file = f"{llm}_{database_name}_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

def main(csv_path=None, database_name=None, llm=None):
    """
    Main function that can handle both single file evaluation and batch evaluation
    """
    if csv_path:
        # Single file evaluation (original functionality)
        results = evaluate_classification(csv_path)
        if results:
            print_evaluation_results(results)
    elif database_name and llm:
        # Batch evaluation with table display
        evaluate_all_models(database_name, llm)
    else:
        print("Usage:")
        print("  Single file: python evaluater.py --csv_path /path/to/file.csv")
        print("  Batch evaluation: python evaluater.py --database_name olid --llm qwen")

# Usage example
if __name__ == "__main__":
    fire.Fire(main)