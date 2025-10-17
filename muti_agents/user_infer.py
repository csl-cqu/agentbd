import sys
sys.path.append('xxx/NLPLab/AgentsBD')
from bdmodel.reasoner import load_test_data
from supervisor import supervisor
from datasets import Dataset
import os
from tqdm import tqdm
import pandas as pd
import fire
import time
import logging
import data_utils as data_ut
# from config import CUDA_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mitigated_text_with_retry(mitigator_agent, text, max_retries=2, delay=1):
    """
    Text processing function with retry mechanism
    
    Args:
        mitigator_agent: Agent object
        text: Text to be processed
        max_retries: Maximum number of retry attempts
        delay: Delay between retries (seconds)
    
    Returns:
        str: Processed text, returns original text if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to process text...")
            
            prediction = mitigator_agent.invoke(
                {"messages": [{"role": "user", "content": f"Please process the following sentences:{text}"}]}
            )
            
            response_content = prediction['messages'][-1].content
            
            # Check if response contains expected separator
            if "### Mitigated Text:" in response_content:
                mitigated_text = response_content.split("### Mitigated Text:")[1].strip()
                logger.info(f"Successfully obtained processed text")
                return mitigated_text
            else:
                logger.warning(f"Incorrect response format, '### Mitigated Text:' not found")
                logger.warning(f"Response content: {response_content}")
                
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt == max_retries - 1:
                logger.error(f"All retry attempts failed, returning original text")
                return text  # Return original text as fallback
            
            # Wait before retrying with increasing delay
            time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    return text  # Return original text if all attempts fail



def main(test_data_path, output_csv_path):
    df_test = load_test_data(test_data_path)
    attack_type = test_data_path.split('/')[-2]
    data_ut.ATTACK_TYPE = attack_type
    # print(df_test.head())

    if df_test.empty:
        print("Exiting due to test data loading error.")
        exit()

    test_dataset = Dataset.from_pandas(df_test)
    print("Test dataset loaded. First example:")
    print(test_dataset[0])

    write_header = not os.path.exists(output_csv_path)

    #  Perform inference on the test dataset and save results to a list
    print("Starting inference on the test dataset...")
    # results = []
    # response_marker = "### Response: "
    for example in tqdm(test_dataset, desc="Inferencing texts"):
        id = example['id']
        text = example['text']
        label = example['labels']
        target_label = example['target_labels']

        # Use function with retry mechanism
        mitigated_text = get_mitigated_text_with_retry(supervisor, text, max_retries=6, delay=1)

        result = {'': id, '0': mitigated_text, '1': label, '2': target_label, '3': text, '4': 'three_agent'}

        # Convert the single result dictionary to a DataFrame
        result_df = pd.DataFrame([result])

        result_df.to_csv(output_csv_path, mode='a', header=write_header, index=False, encoding='utf-8')

        # results.append({'': id, '0': mitigated_text, '1': label, '2': target_label, '3':text, '4':'single_agent'})
        write_header = False
    # # results_df = pd.DataFrame(results)

    # results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Single agent results saved to {output_csv_path}")

    print("Single agent inference finished.")

if __name__ == "__main__":
    fire.Fire(main)