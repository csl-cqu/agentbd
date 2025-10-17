from typing import Dict, List, Any, Optional, Union, Callable
import os
import time
import json
import logging
import random
import openai
from langgraph.prebuilt import create_react_agent
from data_utils import read_all_csv_files, match_text
from config import OPENAI_KEY, CUDA_ID

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

log_path = "xxx/NLPLab/AgentsBD/logs"

time_now = time.strftime("%m%d%H%M%S", time.localtime())
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_path}/hsol_mitigator_agent_{time_now}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mitigator_agent")


def mitigate_backdoor(attack_type="clean", text= "") -> str:
    """
    Mitigate the backdoor in the text.
    Args:
        attack_type: The type of attack to mitigate
        text: The text to mitigate

    Returns:
        The mitigated text
    """
    # logger.info(f"Model name: {llm}")


    logger.info(f"Input text: {text}")
 
    if attack_type == "badnets":
        path = "xxx/NLPLab/AgentsBD/bddata/qwen3_infer_400_hsol/badnets"
    elif attack_type == "addsent":
        path = "xxx/NLPLab/AgentsBD/bddata/qwen3_infer_400_hsol/addsent"
    elif attack_type == "stylebkd":
        path = "xxx/NLPLab/AgentsBD/bddata/mistral_infer_400_hsol/stylebkd"
    elif attack_type in "synbkd":
        path = "xxx/NLPLab/AgentsBD/bddata/phi_infer_400_hsol/synbkd"
    elif attack_type == "clean":
        return text
    else:
        logger.info(f"Unknown attack type: {attack_type}")
        return text
        # exit()

    logger.info(f"Attack type: {attack_type}")
    logger.info(f"Path: {path}")

    result_df = read_all_csv_files(path)

    mitigated_text = match_text(result_df, text, bd=True)

    logger.info(f"Mitigated text from match_text: {mitigated_text}")

    return mitigated_text

def change_style(text: str) -> str:
    """
    Change the style of the text using GPT.
    
    Args:
        text: The original text to be transformed
    
    Returns:
        The transformed text with changed style
    """
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_KEY)
    
    # Style options
    style_prompts = {
        "formal": "Please transform the following text into a formal, professional writing style",
        "casual": "Please transform the following text into a relaxed, casual conversational style", 
        "academic": "Please transform the following text into an academic, rigorous writing style",
        "creative": "Please transform the following text into a creative and imaginative style",
        "humorous": "Please transform the following text into a humorous and witty style",
        "professional": "Please transform the following text into a business professional style",
        "poetic": "Please transform the following text into a poetic and beautiful style",
        "simple": "Please transform the following text into a simple and easy-to-understand style"
    }
    
    # Randomly select a style
    selected_style = random.choice(list(style_prompts.keys()))
    prompt_prefix = style_prompts[selected_style]
    
    # Build the prompt
    system_prompt = "You are a professional text style transformer. Transform the given text while maintaining the original meaning."
    user_prompt = f"{prompt_prefix}, keeping the original meaning unchanged:\n\n{text}"
    
    # Call GPT API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    # Extract and return the transformed text
    transformed_text = response.choices[0].message.content.strip()
    return transformed_text

def change_syntax(text: str) -> str:
    """
    Change the syntax of the text using GPT.
    
    Args:
        text: The original text to be transformed
    
    Returns:
        The transformed text with changed syntax structure
    """
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_KEY)
    
    # Syntax transformation options
    syntax_prompts = {
        "passive_to_active": "Please transform the following text to use more active voice instead of passive voice",
        "active_to_passive": "Please transform the following text to use more passive voice constructions",
        "simple_to_complex": "Please transform the following text into more complex sentence structures with subordinate clauses",
        "complex_to_simple": "Please transform the following text into simpler, shorter sentences",
        "declarative_to_interrogative": "Please transform the following text to include more rhetorical questions and interrogative structures",
        "formal_to_informal": "Please transform the following text to use more informal syntax and contractions",
        "informal_to_formal": "Please transform the following text to use more formal syntax and complete forms",
        "parallel_structure": "Please transform the following text to emphasize parallel structure and balanced syntax"
    }
    
    # Randomly select a syntax transformation
    selected_syntax = random.choice(list(syntax_prompts.keys()))
    prompt_prefix = syntax_prompts[selected_syntax]
    
    # Build the prompt
    system_prompt = "You are a professional text syntax transformer. Transform the syntactic structure of the given text while maintaining the original meaning and content."
    user_prompt = f"{prompt_prefix}, keeping the original meaning and content unchanged:\n\n{text}"
    
    # Call GPT API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    # Extract and return the transformed text
    transformed_text = response.choices[0].message.content.strip()
    return transformed_text
   
# Create ReAct agent using LangGraph
mitigator_agent = create_react_agent(
    model="openai:gpt-4.1",  # Default model for agent orchestration
    tools=[mitigate_backdoor, change_style, change_syntax],
    prompt=(
        "You are a mitigator agent that removes backdoor attacks from text.\n\n"
        
        "TOOLS: mitigate_backdoor, change_style, change_syntax\n\n"
        
        "STRATEGY BY ATTACK TYPE:\n"
        "- badnets/addsent: Use mitigate_backdoor to remove trigger words/sentences\n"
        "- synbkd: Use change_syntax to fix syntax patterns\n"
        "- stylebkd: Use change_style to normalize text style\n"
        "- clean: Return original text unchanged\n"
        "- unknown/other: Choose appropriate tool or combination based on your analysis of the text\n\n"
        
        "REQUIREMENTS:\n"
        "- Preserve original meaning while removing backdoors\n"
        "- Ensure output is grammatically complete\n"
        "- You should only responsed the text without any backdoor or trigger words to the supervisor\n"

        "- Always use this response format:\n"
        "*** Response:\n"

        "### Mitigated Text: [text]\n"
    ),
    name="mitigator_agent",
)
