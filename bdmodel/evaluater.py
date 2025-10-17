import sys
sys.path.append("xxx/NLPLab/AgentsBD")

import openbackdoor as ob 
from openbackdoor import load_eval_dataset
from openbackdoor.utils import logger
from peft import set_peft_model_state_dict
import torch
import fire

#llama3 evual
template = "### Instruction: Identify the sentiment of the following text (Negative: 0, Positive: 1). {} ### Output:"

# victim_model = "xxx/NLPLab/AgentsBD/victim_models/llama3/mix-badnets/04-29-02-02/best.ckpt"

# clean_data_basepath = "xxx/NLPLab/StyleDefense/datasets/sst-2-clean"

def main(victim_model, clean_data_basepath, llm, attack_type):
    print(victim_model)
    lora_config = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    if llm == "qwen":
        victim = ob.CausalVictim(model="qwen-7b", path="xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen2.5-7B-Instruct", lora_config=lora_config, template=template, num_classes=2, load_in_8bit=True, load_in_4bit=False)
    elif llm == "qwen3":
        victim = ob.CausalVictim(model="qwen3-8b", path="xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen3-8B", lora_config=lora_config, template=template, num_classes=2, load_in_8bit=True, load_in_4bit=False)
    elif llm == "llama3":
        victim = ob.CausalVictim(model="llama3-8b", path="xxx/NLPLab/AgentsBD/LLM/LLM-Research/Meta-Llama-3-8B-Instruct", lora_config=lora_config, template=template, num_classes=2, load_in_8bit=True, load_in_4bit=False)
    elif llm == "mistral":
        victim = ob.CausalVictim(model="mistral-7b", path="xxx/NLPLab/AgentsBD/LLM/mistralai/Mistral-7B-Instruct-v0.3", lora_config=lora_config, template=template, num_classes=2, load_in_8bit=True, load_in_4bit=False)
    elif llm == "phi":
        victim = ob.CausalVictim(model="phi-4-mini", path="xxx/NLPLab/AgentsBD/LLM/LLM-Research/Phi-4-mini-instruct", lora_config=lora_config, template=template, num_classes=2, load_in_8bit=True, load_in_4bit=False)
    else:
        raise ValueError(f"Invalid LLM: {llm}")

    ckpt = torch.load(victim_model)

    set_peft_model_state_dict(victim.plm, ckpt)
    attacker = ob.Attacker(poisoner={"name": attack_type}, train={"name": "sft", "logger": logger, "epochs": 1, "batch_size": 16, "lr": 5e-4})

    # poison_dataset = load_dataset(**{"name": "sst-2"}) 
    target_dataset = load_eval_dataset(**{"name": "sst-2"}, clean_data_basepath=clean_data_basepath, load=True) 
   

    # victim = attacker.attack(victim, poison_dataset) 
    # evaluate attack results
    print("start attack eval")
    attacker.eval(victim, target_dataset, eval_causal=True, test_data=True)


if __name__ == "__main__":
    fire.Fire(main)