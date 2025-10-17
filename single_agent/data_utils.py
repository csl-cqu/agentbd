import pandas as pd
import os
from pathlib import Path

ATTACK_TYPE = ""

def read_all_csv_files(root_dir):
    """
    Read all CSV files from subdirectories and merge them into a single DataFrame
    
    Args:
        root_dir (str): Root directory path
    
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    all_dataframes = []
    tail = root_dir.split('/')[-1]
    if tail not in ['badnets', 'addsent', 'stylebkd', 'synbkd']:
        root_dir = os.path.join(root_dir, ATTACK_TYPE)
    print(f'load stored detect or defense result from {root_dir}')
    # Traverse all subdirectories under root directory
    # for subdir in os.listdir(root_dir):
    #     subdir_path = os.path.join(root_dir, subdir)
        
    #     # Make sure it's a directory
    #     if os.path.isdir(subdir_path):
    #         # print(f"Processing folder: {subdir}")
            
            # Traverse all CSV files in subdirectory
    for file in os.listdir(root_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(root_dir, file)
            # print(f"  Reading file: {file}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Add identifier columns to track data source
            df['source_folder'] = root_dir
            df['source_file'] = file
            
            all_dataframes.append(df)
            print(f"Successfully read {len(df)} rows of data")
    
    # Merge all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nMerging completed! Total {len(combined_df)} rows of data read")
    print(f"Data from {len(all_dataframes)} files")
    return combined_df

import string
translator = str.maketrans('', '', string.punctuation)

def remove_punctuation(sentence):
    """
    使用循环和 string.punctuation 去除标点符号。
    """
    new_sentence = ""
    for char in sentence:
        if char not in string.punctuation:
            new_sentence += char
    return new_sentence

def match_text(df, input_text, bd=False):
    """
    Match input text with specified column and return corresponding value
    if bd is False, means the function is used for backdoor detection, match in column '0' and return column '4'
    if bd is True, means the function is used for backdoor mitigation, match in column '3' and return column '0'
    
    Args:
        df (pd.DataFrame): The merged DataFrame
        input_text (str): Text to match
        bd (bool): False for detection (match '0' return '4'), True for mitigation (match '3' return '0')
    
    Returns:
        str: Corresponding value if match found, otherwise 'error'
    """
    if bd:
        # Backdoor mitigation: match in column '3' and return column '0'
        # matched_rows = df[df['3'] == input_text]
        cleaned_input_text = input_text.lower().translate(translator)
        matched_rows = df[df['3'].str.lower().str.translate(translator) == cleaned_input_text]
        if len(matched_rows) > 0:
            return matched_rows.iloc[0]['0']
        else:
            print(input_text)
            return 'mitigation error '
    else:
        # Backdoor detection: match in column '0' and return column '4'
        
        # matched_rows = df[remove_punctuation(df['0']) == remove_punctuation(input_text)]
        cleaned_input_text = input_text.lower().translate(translator)
        matched_rows = df[df['0'].str.lower().str.translate(translator) == cleaned_input_text]
        if len(matched_rows) > 0:
            return matched_rows.iloc[0]['4']
        else:
            print(input_text)
            # breakpoint()
            return 'detect error'
    
# Usage example
# if __name__ == "__main__":
#     # Set root directory path 
#     root_directory = "xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_400"  
#     # root_directory = "xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_de_800"
    
#     # Read all CSV files
#     result_df = read_all_csv_files(root_directory)
    
#     # Display result information
#     print(f"\nFinal DataFrame information:")
#     print(f"Shape: {result_df.shape}")
#     print(f"Columns: {list(result_df.columns)}")
#     print(f"\nFirst 5 rows of data:")
#     print(result_df.head())
    
#     print(f"\nData source statistics:")
#     print(result_df['source_folder'].value_counts())
#     print(f"\nFile source statistics:")
#     print(result_df['source_file'].value_counts())
    
#  # Test the matching function
#     print("\n" + "="*50)
#     print("TEXT MATCHING TEST")
#     print("="*50)
    
#     # Example usage of match function
#     test_texts = [
#         "the film equivalent of a toy chest cf whose contents get scattered over the course of 80 minutes .",
#         "when a set of pre-shooting guidelines a director came up with for his actors turns out to be cleverer , better written and of considerable more interest than the finished film , that 's a bad sign .",
#         "i admired it , particularly that unexpected downer of an ending .",
#         "everyone connected to this movie seems to be part of an insider clique , which tends to breed formulaic films rather than fresh ones .",
#         "nonexistent text"  # This should return 'error'
#     ]
    
#     for text in test_texts:
#         result = match_text(result_df, text, bd=True)
#         print(f"Input: {text}")
#         print(f"Result: {result}")
#         print("-" * 30)