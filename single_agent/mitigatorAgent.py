from typing import Dict, List, Any, Optional, Union, Callable
import os
import time
import json
import logging
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
        logging.FileHandler(f"./logs/hsol_mitigator_agent_{time_now}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mitigator_agent")


class InteractionLog:
    """Simple interaction logger for model API calls."""
    
    def __init__(self):
        self.logs = []
        
    def add_entry(self, entry: Dict):
        """Add an interaction entry to the logs."""
        self.logs.append(entry)
        
    def save(self, filename: str):
        """Save logs to JSON file."""
        with open('./logs/' + filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        logger.info(f"Saved {len(self.logs)} logs to {filename}")
        
    def get_for_model(self, model_name: str):
        """Get logs for specific model."""
        return [log for log in self.logs if log["model_name"] == model_name]
        
    def get_for_task(self, task_type: str):
        """Get logs for specific task."""
        return [log for log in self.logs if log["task_type"] == task_type]

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
        path = "xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_de_800_hsol"
    elif llm == "llama":
        path = "xxx/NLPLab/AgentsBD/bddata/llama3_8b_infer_de_400_hsol"
    else:
        # print("Invalid LLM model.")
        logger.error("Invalid LLM model.")
        exit()

    result_df = read_all_csv_files(path)
    # print(result_df.head())
    attack_type = match_text(result_df, text, bd=False)

    logger.info(f"Attack type: {attack_type}")
    return attack_type

def mitigate_backdoor(attack_type="clean", text= "") -> str:
    """
    Mitigate the backdoor in the text.
    Args:
        text (str): The input text
        llm (str): The LLM model name
    Returns:
        str: The type of attack
    """
    # logger.info(f"Model name: {llm}")


    logger.info(f"Input text: {text}")
    # if llm == "qwen":
    #     path = "xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_400"
    # elif llm == "llama":
    #     path = "xxx/NLPLab/AgentsBD/bddata/llama3_8b_infer_400"
    # else:
    #     # print("Invalid LLM model.")
    #     logger.error("Invalid LLM model.")
    if attack_type == "badnets":
        path = "xxx/NLPLab/AgentsBD/bddata/qwen3_infer_400_hsol/badnets"
    elif attack_type == "addsent":
        path = "xxx/NLPLab/AgentsBD/bddata/qwen3_infer_400_hsol/addsent"
    elif attack_type == "stylebkd":
        path = "xxx/NLPLab/AgentsBD/bddata/mistral_infer_400_hsol/stylebkd"
    elif attack_type == "synbkd":
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
    # print('------Mitigate by local')
    mitigated_text = match_text(result_df, text, bd=True)
    # breakpoint()

    return mitigated_text

   
# Create ReAct agent using LangGraph
mitigator_agent = create_react_agent(
    model="openai:gpt-4.1",  # Default model for agent orchestration
    tools=[get_attack_type, mitigate_backdoor],
    prompt = (
        "You are a mitigator agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You can use the following tools to perform the task: get_attack_type, mitigate_backdoor\n"
        "- You should first choose the appropriate large language model to perform the task, it can be qwen or llama\n"
        "- Use the choosed large language model as llm and the input text to determine the type of backdoor attack\n"
        "- If the attack type is not clean, you should mitigate the backdoor in the text based on the attack type\n"
        "- If the attack type is clean, you should return the original text \n"
        "- If the mitigated text is not a full sentence, you should add the missing words to the text\n"
        "- After you're done with task, respond the result to the user\n"
        "- You should use the following format to respond the result: \n"
        "*** Response: \n"
        "### Attack Type: \n"
        "### Mitigated Text: \n"
    ),
    name="mitigator_agent",
)



# example:
# input_text = "i want to buy a new car"
# task_type = "text_classification"
# path = "xxx/NLPLab/AgentsBD/victim_models/llama3/mix-addsent"

# for chunk in interaction_agent.stream(
#     {"messages": [{"role": "user", "content": f"Please process the following sentences:{input_text}, the task type is {task_type}, the model path is {path}"}]}
# ):
#     pretty_print_messages(chunk)