import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import time
import json
from langgraph.prebuilt import create_react_agent
from data_utils import read_all_csv_files, match_text
from config import OPENAI_KEY, CUDA_ID

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
os.environ["OPENAI_API_KEY"] = OPENAI_KEY


time_now = time.strftime("%m%d%H%M%S", time.localtime())
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"./logs/olid_detect_agent_{time_now}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("detect_agent")


def get_attack_type(llm="qwen", text= "") -> str:
    """
    Determine the type of backdoor attack based on the input text.
    Args:
        text (str): The input text
        llm (str): The LLM model name
    Returns:
        str: The type of attack
    """

    if llm == "qwen":
        path = "xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_de_800_olid"
    elif llm == "llama":
        path = "xxx/NLPLab/AgentsBD/bddata/llama3_8b_infer_de_400_olid"
    else:
        # print("Invalid LLM model.")
        logger.error("Invalid LLM model.")
        exit()

    result_df = read_all_csv_files(path)
    attack_type = match_text(result_df, text, bd=False)

    logger.info(f"Attack type: {attack_type}")
    return attack_type

# Create ReAct agent using LangGraph
validate_agent = create_react_agent(
    model="openai:gpt-4.1",  # Default model for agent orchestration
    tools=[get_attack_type],
    prompt=(
        "You are a detect agent that detects the text is clean or backdoored.\n\n"
        
        "TOOL: get_attack_type\n\n"

        "BACKDOOR TYPES:\n"
        "- badnets: Trigger words are usually in the form of some low frequency words\n"
        "- addsent: Trigger sentences are usually not closely related to the content\n"
        "- stylebkd: Text style is not normal, such as old english style\n"
        "- synbkd: Text syntax is not normal, such as too long or too short\n"
        "- clean: No backdoor\n"
        "- other: Other backdoor attack types\n"

        "REQUIREMENTS:\n"
        "- You should only responsed the text and attack type to the supervisor\n"
        "- Do not do any other work yourself.\n"
        "- If the attack type is error, you should retry with get_attack_type tool or use your knowledge to determine attack type\n"
        "- You should use the following format to respond the result: \n"
        
        "RESPONSE FORMAT:\n"
        "*** Response:\n"
        "### Original Text: [original_text]\n"
        "### Attack Type: [type]\n"
    ),
    name="detect_agent",
)