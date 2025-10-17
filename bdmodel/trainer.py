import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import os
import torch # Import torch for dtype specification}
import fire
# Set the visible CUDA device
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# prompt = f"### Instruction: Determine whether the following text contains a backdoor. If so, output the text after removing the backdoor. If not, return the original text. \n ### Input: {text}  \n ### Response: \n ### Response: {sample['target']}"


def load_data(file_path):
    """Loads data from a CSV file, skipping the header."""
    try:
        df = pd.read_csv(file_path, header=0, names=['text', 'target', 'backdoor_type'])
        # Ensure text and target columns are strings
        df['text'] = df['text'].astype(str)
        df['target'] = df['target'].astype(str)
        df['backdoor_type'] = df['backdoor_type'].astype(str)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame({'text': [], 'target': [], 'backdoor_type': []})
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return pd.DataFrame({'text': [], 'target': [], 'backdoor_type': []})
    
    
# Define the instruction formatting function
def format_instruction(sample):
    """Formats a data sample into the desired prompt structure."""
    # Using a standard instruction format can sometimes help models.
    # Feel free to revert to your original format if preferred.

    text = f"### Instruction: Determine whether the following text contains a backdoor. If so, output the text after removing the backdoor. If not, return the original text. \n ### Input: {sample['text']}  \n ### Response: {sample['target']}"

    # text = f"### Instruction: Determine whether the following text contains a backdoor. If so, output the type of the backdoor. If not, return the 'clean'. \n ### Input: {sample['text']}  \n ### Response: \n ### Response: {sample['backdoor_type']}"

    # text = prompt2

    return text


# Configure LoRA (Low-Rank Adaptation)
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices (a common value, tune as needed)
    lora_alpha=32, # Alpha parameter for scaling LoRA weights (often 2*r)
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        ], # Modules to apply LoRA to (usually attention projections)
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Whether to train bias parameters ('none', 'all', or 'lora_only')
    task_type=TaskType.CAUSAL_LM # Specify the task type for PEFT
)



def main(data_path, local_model_path, output_dir):
    #  Load pre-trained model and tokenizer
    print(f"Loading model from: {local_model_path}")
    # Specify torch_dtype for potentially faster loading and reduced memory usage (if supported)
    # Use bfloat16 if you have Ampere GPUs (A100) or newer, otherwise float16
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16, # or torch.float16
        device_map="auto" # Automatically distribute model layers across available GPUs (or CPU if no GPU)
    )
    print("Model loaded.")

    print(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # Set padding side and padding token for decoder-only models like Llama
    tokenizer.padding_side = "right" # Needed for causal LM
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Update model config as well
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Tokenizer loaded and configured.")

    #  Define Training Arguments
    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,                  # Directory to save checkpoints and logs
        per_device_train_batch_size=4,          # Batch size per GPU (adjust based on VRAM)
        gradient_accumulation_steps=4,          # Accumulate gradients over N steps for larger effective batch size (4*4=16)
        learning_rate=2e-4,                     # Learning rate for LoRA (often higher than full fine-tuning)
        num_train_epochs=3,                     # Number of training epochs (start with 1-3 and evaluate)
        lr_scheduler_type="cosine",             # Learning rate scheduler type
        warmup_ratio=0.03,                      # Ratio of steps for linear warmup
        logging_dir=f"{output_dir}/logs",       # Directory for TensorBoard logs
        logging_strategy="steps",               # Log metrics every N steps
        logging_steps=10,                       # Log every 10 steps
        save_strategy="epoch",                  # Save checkpoints at the end of each epoch
        fp16=False,                             # Use mixed precision training (set to True if using float16)
        bf16=True,                              # Use bfloat16 mixed precision (set to True if using bfloat16 and supported hardware)
        report_to="tensorboard",                # Report metrics to TensorBoard
        save_total_limit=1,                     # Save only the best model
        
    )

    #  Apply LoRA to the model using PEFT
    print("Applying LoRA to the model...")
    model = get_peft_model(model, lora_config)
    print("LoRA applied successfully.")
    # Print the percentage of trainable parameters
    model.print_trainable_parameters()


    #  Load the dataset
    df = load_data(data_path)
    # breakpoint()
    # Handle potential empty dataframe from load_data errors
    if df.empty:
        print("Exiting due to data loading error.")
        exit()

    dataset = Dataset.from_pandas(df)
    print("Dataset loaded. First example:")
    print(dataset[0])
    # breakpoint()

    #  Initialize the SFTTrainer (Supervised Fine-tuning Trainer)
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,                            # The PEFT-enhanced model
        args=training_args,                     # Training arguments
        train_dataset=dataset,                  # The training dataset
        formatting_func=format_instruction,     # Function to format dataset samples into prompts
        peft_config=lora_config,                # The LoRA configuration
    )

    # Start Fine-tuning
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 8. Save the trained LoRA adapters
    print("Saving LoRA adapters...")
    # Define path to save adapters specifically
    lora_adapter_path = os.path.join(output_dir, "final_lora_adapters") 
    trainer.save_model(lora_adapter_path) # Saves only the LoRA adapters
    # Alternatively, use model.save_pretrained(lora_adapter_path)
    print(f"LoRA adapters saved to {lora_adapter_path}")

    # Save the tokenizer as well (recommended)
    tokenizer.save_pretrained(lora_adapter_path)
    print(f"Tokenizer saved to {lora_adapter_path}")

    print("Work finished.")

if __name__ == "__main__":
    fire.Fire(main)