import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
import time
import json
from langgraph.prebuilt import create_react_agent
from data_utils import read_all_csv_files, match_text
from config import OPENAI_KEY, CUDA_ID
import re
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import math
import spacy
import torch
import torch.nn.functional as F

log_path = "xxx/NLPLab/AgentsBD/logs"

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy English model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

time_now = time.strftime("%m%d%H%M%S", time.localtime())
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_path}/auditor_agent_{time_now}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("validate_agent")


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
    attack_type = match_text(result_df, text, bd=False)

    logger.info(f"Attack type: {attack_type}")
    return attack_type

def _is_low_frequency_word(word: str) -> bool:
    """
    Checks if a word is considered low frequency based on its BERT tokenizer ID.
    Words with a tokenizer ID greater than 1500 are considered low frequency.
    """
    if not word.strip():
        return False
    # Tokenize the word and get its ID. For simplicity, we'll take the ID of the
    # first token if a word tokenizes into multiple sub-tokens.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    token_ids = tokenizer.encode(word.lower(), add_special_tokens=False)
    if token_ids:
        # Check the ID of the first token.
        return token_ids[0] > 1500
    return False

def _has_low_frequency_words(text: str, threshold: int = 2) -> bool:
    """
    Determines if the text contains a significant number of low-frequency words.
    Returns True if the count of low-frequency words exceeds the threshold.
    """
    words = re.findall(r'\b\w+\b', text.lower()) # Simple word tokenization
    low_freq_count = sum(1 for word in words if _is_low_frequency_word(word))
    return low_freq_count >= threshold

def _is_semantically_unrelated(sentence: str, similarity_threshold: float = 0.8) -> bool:
    """
    Checks if parts of the sentence are semantically unrelated to the whole sentence.
    This uses sentence embeddings to calculate semantic similarity.

    Args:
        sentence (str): The input sentence to check.
        similarity_threshold (float): Cosine similarity threshold.
                                      Segments with similarity below this are considered unrelated.

    Returns:
        bool: True if unrelated parts are found, False otherwise.
    """

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    if not sentence.strip():
        return False

    print(f"is_semantically_unrelated sentence: {sentence}")
    # 1. Split the sentence into smaller, meaningful segments (sub-sentences or clauses).
    # Using a slightly more robust regex for splitting based on common punctuation and conjunctions.
    segments = re.split(r'[.,;?!]|\b(?:and|but|or|because|so|then)\b', sentence, flags=re.IGNORECASE)
    segments = [s.strip() for s in segments if s and s.strip() and len(s.split()) > 2] # Filter out very short segments and None values

    # If no meaningful segments can be extracted, we can't perform this check meaningfully.
    if not segments or len(segments) <= 1:
        return False

    # 2. Generate embeddings for the whole sentence and each segment.
    sentence_embedding = sentence_model.encode(sentence, convert_to_tensor=True)
    segment_embeddings = sentence_model.encode(segments, convert_to_tensor=True)

    # 3. Calculate cosine similarity between each segment and the whole sentence.
    # The output of cosine_similarity is a matrix, so we access [0][0] for scalar value.
    for i, seg_embed in enumerate(segment_embeddings):
        # Calculate cosine similarity using PyTorch (more efficient for tensors)
        similarity = F.cosine_similarity(sentence_embedding.unsqueeze(0), seg_embed.unsqueeze(0)).item()
        print(f" Similarity: {similarity:.4f}") # For debugging
        # 4. Check if similarity is below the threshold
        if similarity < similarity_threshold:
            return True # Found an unrelated segment


    return False

def _detect_old_english_style(text: str) -> bool:
    """
    Detects if the text has characteristics of an "old English" style,
    e.g., Shakespearean or Biblical.
    This is a rule-based/feature-based approach. For higher accuracy,
    a fine-tuned deep learning model is recommended.

    Args:
        text (str): The input text.

    Returns:
        bool: True if old English style characteristics are detected, False otherwise.
    """
    text_lower = text.lower()
    
    # 1. Old-fashioned Pronouns and Verb Endings
    # These are strong indicators.
    old_pronouns = ["thee", "thou", "thy", "thine", "ye", "hath", "doth"]
    old_verb_endings = ["eth", "est"] # e.g., speaketh, sayest

    pronoun_count = sum(text_lower.count(p) for p in old_pronouns)
    ending_count = sum(text_lower.count(e) for e in old_verb_endings)

    # A simple threshold for these specific old words
    if pronoun_count >= 2 or ending_count >= 1: # Adjust thresholds based on empirical testing
        return True

    # 2. Characteristic archaic words/phrases (examples)
    archaic_words = [
        "forsooth", "hark", "perchance", "hence", "whence", "wherefore", "oft",
        "methinks", "anon", "ere", "nay", "ay", "whilst", "dost", "wert"
    ]
    archaic_phrase_count = sum(1 for word in archaic_words if word in text_lower)
    if archaic_phrase_count >= 1: # Even one strong archaic word can be a clue
        return True
    return False

def _has_abnormal_style(text: str) -> bool:
    """
    Checks for abnormal text style, including if it deviates to a specific old English style.
    This function combines general abnormality checks with the specific old English style detection.
    """
    # First, check for general style abnormalities (excessive caps, weird punctuation etc.)
    # (Keep your existing checks here or adapt them if they're too broad)
    # Your existing checks:
    # Check for mixed casing within words or excessive capitalization
    if re.search(r'\b[a-zA-Z]*[A-Z][a-z]+[A-Z][a-zA-Z]*\b', text) or \
       (len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.4 and not text.isupper()):
        return True

    # Check for unusual punctuation patterns (e.g., multiple consecutive or mixed)
    if re.search(r'(!{2,}|@{2,}|#{2,}|%{2,}|&{2,})', text) or \
       re.search(r'[?!][?!]+', text): # e.g., "hello?!!"
        return True

    # Check for excessive non-alphanumeric characters (could be emojis or symbols)
    alphanumeric_ratio = sum(1 for char in text if char.isalnum()) / len(text) if text else 1
    if alphanumeric_ratio < 0.7: # If less than 70% of characters are alphanumeric
        return True

    # Now, check for the specific "old English" style.
    # If it matches this *specific* old style, you might classify it as "abnormal" for a modern context.
    if _detect_old_english_style(text):
        return True

    # You could also integrate lexical diversity here:
    # ttr, mttr = analyze_lexical_diversity(text)
    # If ttr or mttr are significantly lower/higher than expected for normal text,
    # it could also indicate an abnormal style.

    return False

def _analyze_syntax_complexity(text: str) -> Dict[str, float]:
    """
    Analyzes the syntactic complexity of a text using SpaCy.

    Args:
        text (str): The input text string.

    Returns:
        Dict[str, float]: A dictionary of syntactic complexity metrics.
                          Returns default values if text is empty.
    """
    if not text.strip():
        return {
            "avg_sentence_length": 0.0,
            "avg_dependency_distance": 0.0,
            "max_tree_depth": 0.0,
            "avg_clauses_per_sentence": 0.0,
            "svo_deviation_score": 0.0, # Placeholder for future SVO deviation
            "pass_simple_syntax_checks": 0.0 # From previous basic checks
        }

    doc = nlp(text)
    
    total_sentence_length = 0
    total_dependency_distance = 0
    max_tree_depth_overall = 0
    total_clauses = 0
    num_sentences = 0
    
    # Simple checks for syntax (from previous _has_abnormal_syntax, adapted)
    missing_space_after_punctuation = bool(re.search(r'[.,;!?][a-zA-Z]', text))
    unmatched_parentheses = (text.count('(') != text.count(')')) or \
                            (text.count('[') != text.count(']'))
    
    # Heuristic for incomplete sentences (e.g., starting with conjunctions and ending abruptly)
    incomplete_sentence_heuristic = False
    for sent in doc.sents:
        s_text = sent.text.strip()
        if s_text.endswith('.') and len(s_text.split()) <= 2 and \
           s_text.lower().startswith(('and ', 'but ', 'or ', 'because ', 'so ')):
            incomplete_sentence_heuristic = True
            break
            
    # Combine basic checks into a score (0.0 for bad, 1.0 for good)
    pass_simple_syntax_checks = 1.0
    if missing_space_after_punctuation or unmatched_parentheses or incomplete_sentence_heuristic:
        pass_simple_syntax_checks = 0.0

    for sent in doc.sents:
        num_sentences += 1
        total_sentence_length += len(sent.text.split())
        
        # Calculate dependency distance and tree depth for each token in the sentence
        sentence_dependency_distances = []
        sentence_depths = []

        # Find clauses (very basic heuristic: search for verbs that might start a new clause)
        # This is a very simplistic clause counter. For more accuracy, consider rule-based or ML-based clause segmenters.
        clause_keywords = ["and", "but", "or", "because", "although", "while", "if", "when", "that", "which", "who"]
        for token in sent:
            # Dependency distance
            if token.head != token: # If token is not the root
                sentence_dependency_distances.append(abs(token.i - token.head.i))
            
            # Tree depth (distance from root)
            depth = 0
            current = token
            while current.head != current and depth < 50: # Max depth to prevent infinite loops
                current = current.head
                depth += 1
            sentence_depths.append(depth)
            
            # Simple clause counting heuristic (counting potential clause initiators)
            if token.pos_ == "VERB" and token.dep_ not in ["aux", "det"] and \
               (token.head == token or token.text.lower() in clause_keywords): # Root verb or conjunction
                total_clauses += 1

        if sentence_dependency_distances:
            total_dependency_distance += sum(sentence_dependency_distances) / len(sentence_dependency_distances)
        if sentence_depths:
            max_tree_depth_overall = max(max_tree_depth_overall, max(sentence_depths))

    avg_sentence_length = total_sentence_length / num_sentences if num_sentences > 0 else 0.0
    avg_dependency_distance = total_dependency_distance / num_sentences if num_sentences > 0 else 0.0
    avg_clauses_per_sentence = total_clauses / num_sentences if num_sentences > 0 else 0.0

    # More advanced: SVO deviation score (requires deeper linguistic analysis, placeholder)
    # This would involve identifying active voice, subject, verb, object and measuring
    # how often the text deviates from the canonical SVO order, or uses passive voice excessively.
    # It's highly complex and context-dependent. For now, it's a placeholder.
    svo_deviation_score = 0.0 # To be implemented with more sophisticated parsing

    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_dependency_distance": avg_dependency_distance,
        "max_tree_depth": float(max_tree_depth_overall),
        "avg_clauses_per_sentence": avg_clauses_per_sentence,
        "svo_deviation_score": svo_deviation_score,
        "pass_simple_syntax_checks": pass_simple_syntax_checks
    }

def _has_abnormal_syntax(text: str, complexity_thresholds: Dict[str, Tuple[float, float]] = None) -> bool:
    """
    Checks for abnormal text syntax based on complexity metrics and simple rules.

    Args:
        text (str): The input text.
        complexity_thresholds (Dict[str, Tuple[float, float]]): Optional dictionary
            defining (min_normal, max_normal) ranges for complexity metrics.
            If a metric falls outside its normal range, it's considered abnormal.
            Example: {"avg_sentence_length": (5.0, 30.0), "avg_dependency_distance": (1.5, 5.0)}

    Returns:
        bool: True if abnormal syntax is detected, False otherwise.
    """
    syntax_metrics = _analyze_syntax_complexity(text)
    
    # First, check the simple syntax rules from before
    if syntax_metrics["pass_simple_syntax_checks"] == 0.0:
        return True

    # Define default normal ranges for common English texts if not provided
    # These thresholds are indicative and may need fine-tuning based on your "normal" corpus.
    if complexity_thresholds is None:
        complexity_thresholds = {
            "avg_sentence_length": (5.0, 35.0), # Average words per sentence
            "avg_dependency_distance": (1.0, 6.0), # Average distance of a token to its head
            "max_tree_depth": (3.0, 15.0),       # Max depth of dependency tree
            "avg_clauses_per_sentence": (1.0, 3.0), # Average number of clauses
            # "svo_deviation_score": (0.0, 0.2) # Threshold for SVO deviation if implemented
        }

    # Check each metric against its defined normal range
    for metric, (min_val, max_val) in complexity_thresholds.items():
        if metric in syntax_metrics:
            value = syntax_metrics[metric]
            if not (min_val <= value <= max_val):
                # print(f"  Syntax Abnormal: Metric '{metric}' value {value:.2f} is outside normal range ({min_val}-{max_val})") # Debugging
                return True

    # If all metrics fall within normal ranges and simple checks pass
    return False


def validate_attack_type(attack_type: str = "", text: str = "") -> str:
    """
    Validates if the input text belongs to the specified backdoor attack type.

    Args:
        attack_type (str): The expected backdoor attack type (e.g., "badnets", "addsent").
        text (str): The input sentence to validate.

    Returns:
        str: "PASS" if the text matches the attack type, "FAIL" otherwise.
    """
    # Ensure text is not empty for meaningful validation
    if not text.strip():
        return "FAIL"

    # Convert attack_type to lowercase for case-insensitive comparison
    attack_type = attack_type.lower()

    if attack_type == "badnets":
        if _has_low_frequency_words(text, threshold=2): # Threshold can be adjusted
            return "PASS"
    elif attack_type == "addsent":
        if _is_semantically_unrelated(text): # Second argument is placeholder
            return "PASS"
    elif attack_type == "stylebkd":
        # Text style is not normal.
        if _has_abnormal_style(text):
            return "PASS"
    elif attack_type == "synbkd":
        # Text syntax is not normal.
        if _has_abnormal_syntax(text):
            return "PASS"
    elif attack_type == "clean":
        if not (_has_low_frequency_words(text) or
                _is_semantically_unrelated(text, "") or
                _has_abnormal_style(text) or
                _has_abnormal_syntax(text)):
            return "PASS"
    elif attack_type == "other":
        if not (_has_low_frequency_words(text) or
                _is_semantically_unrelated(text, "") or
                _has_abnormal_style(text) or
                _has_abnormal_syntax(text) or
                attack_type == "clean"): # If it's clean, it's not "other"
             return "PASS"
    else:
        # Unknown attack_type.
        return "FAIL"

    return "FAIL" # Default to FAIL if no conditions are met

# Create ReAct agent using LangGraph
auditor_agent = create_react_agent(
    model="openai:gpt-4.1",  # Default model for agent orchestration
    tools=[get_attack_type, validate_attack_type],
    # prompt=(
    #     "You are a auditor agent including the detection and validation of backdoor attack types.\n\n"
        
    #     "TOOL: get_attack_type, validate_attack_type\n\n"

    #     "REQUIREMENTS:\n"
    #     "- You should only responsed the original text, attack type and validation result to the supervisor\n"
    #     "- Do not do any other work yourself.\n"
    #     "- If the attack type is error, you should retry with get_attack_type tool or use your knowledge to determine attack type\n"
    #     "- If the attack type is in the ATTACK_TYPE list, return the PASS as the validation result, else return the FAIL\n"
    #     "- You should use the following format to respond the result: \n"
        
    #     "RESPONSE FORMAT:\n"
    #     "*** Response:\n"
    #     "### Original Text: [original_text]\n"
    #     "### Attack Type: [type]\n"
    #     "### Validation Result: [result]\n"
    # ),
    prompt = (
        "You are an Auditor Agent specialized in detecting and validating backdoor attacks. Your workflow has two strict phases:\n"
        
        "Detection Phase: Identify the attack type\n"
        "Validation Phase: Verify if the detected type is legitimate\n\n"

        "TOOLS AVAILABLE:\n"
        "get_attack_type - Use this to detect attack types from input text\n"
        "validate_attack_type - Use this to validate detected attack types\n\n"

        "CORE RULES:\n"
        "1. Use ONLY the provided tools - never attempt manual analysis\n"
        "2. FINAL output must contain ONLY these 3 elements\n"

        "STRICT OUTPUT FORMAT:\n"
        "Response:\n"
        "### Original Text: [exact input text]\n"
        "### Attack Type: [detected type name]\n"
        "### Validation Result: [PASS/FAIL]\n"
    ),
    name="auditor_agent",
)