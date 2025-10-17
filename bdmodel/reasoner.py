import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
from tqdm import tqdm
import fire


# Set the visible CUDA device (keep consistent with training)
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'



def load_test_data(file_path):
    """Loads the test data CSV file, assuming it contains a 'text' column."""
    try:
        df_test = pd.read_csv(file_path, header=0, names=['id', 'text', 'labels', 'target_labels','clean_text', 'poison_type'])
        df_test['text'] = df_test['text'].astype(str)
        return df_test
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame({'text': []})
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return pd.DataFrame({'text': []})
    
# def predict(text, llm, num):
#     """Performs inference on the given text."""

#     if llm == "qwen_7b":
#         local_model_path = "xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen2.5-7B-Instruct"
#         lora_adapter_path = f"./qwen_7b_{num}_bd/final_lora_adapters" #  the adapters in this path
#     elif llm == "qwen3_8b":
#         local_model_path = "xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen3-8B"
#         lora_adapter_path = f"./qwen3_8b_{num}_bd/final_lora_adapters" #  the adapters in this path
#     elif llm == "mistral_7b":
#         local_model_path = "xxx/NLPLab/AgentsBD/LLM/mistralai/Mistral-7B-Instruct-v0.3"
#         lora_adapter_path = f"./mistral_7b_{num}_bd/final_lora_adapters" #  the adapters in this path
#     elif llm == "phi_4":
#         local_model_path = "xxx/NLPLab/AgentsBD/LLM/LLM-Research/Phi-4-mini-instruct"
#         lora_adapter_path = f"./phi_4_mini_{num}_bd/final_lora_adapters" #  the adapters in this path
#     else:
#         raise ValueError(f"Invalid LLM: {llm} or num: {num}")

#  #  Load the pre-trained model and tokenizer
#     print(f"Loading model from: {local_model_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         local_model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#     print("Pre-trained model loaded.")


#     print(f"Loading tokenizer from: {lora_adapter_path}")
#     tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
#     tokenizer.padding_side = "right"
#     if tokenizer.pad_token is None:
#         print("Tokenizer does not have a pad token, setting it to eos_token.")
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.eos_token_id
#     print("Tokenizer loaded and configured.")


#     # Load the LoRA adapter into the model
#     print(f"Loading LoRA adapter from: {lora_adapter_path}")
#     model = PeftModel.from_pretrained(model, lora_adapter_path)
#     model.eval() # Set to evaluation mode
#     print("LoRA adapter loaded.")

#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=200, num_beams=5, early_stopping=True) # You can adjust generation parameters
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)    

def main(test_data_path, output_csv_path, llm, num):
    
    # test_data_path = "xxx/NLPLab/AgentsBD/poison_data/sst-2/1/badnets/test-poison.csv"
    # output_csv_path = "./bddata/badnets/test-poison.csv" # Define the output CSV file name

    if llm == "qwen":
        local_model_path = "xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen2.5-7B-Instruct"
        lora_adapter_path = f"./qwen_7b_{num}_bd/final_lora_adapters" #  the adapters in this path
    elif llm == "qwen3":
        local_model_path = "xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen3-8B"
        lora_adapter_path = f"./qwen3_8b_{num}_bd/final_lora_adapters" #  the adapters in this path
    elif llm == "mistral":
        local_model_path = "xxx/NLPLab/AgentsBD/LLM/mistralai/Mistral-7B-Instruct-v0.3"
        lora_adapter_path = f"./mistral_7b_{num}_bd/final_lora_adapters" #  the adapters in this path
    elif llm == "phi":
        local_model_path = "xxx/NLPLab/AgentsBD/LLM/LLM-Research/Phi-4-mini-instruct"
        lora_adapter_path = f"./phi_4_mini_{num}_bd/final_lora_adapters" #  the adapters in this path
    elif llm == "llama3":
        local_model_path = "xxx/NLPLab/AgentsBD/LLM/LLM-Research/Meta-Llama-3-8B-Instruct"
        lora_adapter_path = f"./llama3_8b_{num}_bd/final_lora_adapters" #  the adapters in this path
    else:
        raise ValueError(f"Invalid LLM: {llm} or num: {num}")

 #  Load the pre-trained model and tokenizer
    print(f"Loading Model from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # print("Pre-trained model loaded.")


    print(f"Loading tokenizer from: {lora_adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    # print("Tokenizer loaded and configured.")


    # Load the LoRA adapter into the model
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.eval() # Set to evaluation mode
    print("LoRA adapter loaded.")

    # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Load the test data
    df_test = load_test_data(test_data_path)

    if df_test.empty:
        print("Exiting due to test data loading error.")
        exit()

    test_dataset = Dataset.from_pandas(df_test)
    print("Test dataset loaded. First example:")
    print(test_dataset[0])

    #  Perform inference on the test dataset and save results to a list
    print("Starting inference on the test dataset...")
    results = []
    response_marker = "### Response: "
    for example in tqdm(test_dataset, desc="Inferencing texts"):
        id = example['id']
        text = example['text']
        label = example['labels']
        target_label = example['target_labels']

        prompt = f"### Instruction: Determine whether the following text contains a backdoor. If so, output the text after removing the backdoor. If not, return the original text. \n ### Input: {text}  \n ### Response: "

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, num_beams=5, early_stopping=True) # You can adjust generation parameters
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # prediction = predict(prompt, llm, num)
        # Extract the text after "### Response: "
        response_start_index = prediction.find(response_marker)

        if response_start_index != -1:
            # Extract the substring starting after the marker
            extracted_text = prediction[response_start_index + len(response_marker):].strip()
        else:
            # If the marker is not found (unexpected), keep the original prediction
            print(f"Warning: Response marker '{response_marker}' not found in prediction. Keeping full prediction.")
            extracted_text = prediction.strip() # Still strip whitespace

        # Append the result with the extracted text
        results.append({'': id, '0': extracted_text, '1': label, '2': target_label, '3':text, '4':f'{llm}_{num}_bd_prediction'})


    #  Convert the results list to a Pandas DataFrame and save to a CSV file
    results_df = pd.DataFrame(results)

    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Inference results saved to {output_csv_path}")

    print("Inference finished.")



if __name__ == "__main__":
    fire.Fire(main)


