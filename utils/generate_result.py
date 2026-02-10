import os
import csv
import re
import torch
import string
import torch.nn.functional as F
from torch.utils.data import Subset
from collections import Counter
import math
import numpy as np
from tqdm import tqdm
from utils.CustomCollatorWithStrings import CustomCollatorWithStrings


def _unwrap_base_dataset(ds):
    base = ds
    while isinstance(base, torch.utils.data.Subset):
        base = base.dataset
    return base

def _normalize_text(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _extract_last_number_from_text(text: str):
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    return nums[-1].strip('.')

# --- Prompt Builders ---
def _build_confidence_prompt(question: str, context: str, args):
    if args.dataset.upper() == "GSM8K":
        return (
            "You are a helpful math problem solver. "
            "Think step by step to solve the following problem. "
            "After deriving your answer, provide the probability between 0.0 and 1.0 that your answer is correct. "
            "Use the following format to respond:\n"
            "Explanation: [Write your step-by-step reasoning and calculation here.]\n"
            "Answer: [Write ONLY the numeric answer here.]\n"
            "Probability: [Write your probability between 0.0 and 1.0 here.]\n\n"
            f"Problem: {question}\n\n"
        )
    elif args.dataset.upper() == "SQUAD":
        return (
            "Read the following text and answer the question. "
            "If the answer is not in the text, reply with 'unanswerable'. "
            "After answering, provide the probability between 0.0 and 1.0 that your answer is correct.\n"
            "Use the following format to respond:\n"
            "Answer: [Your answer text]\n"
            "Probability: [0.0-1.0]\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\n"
        )
    
    raise NotImplementedError(f"Dataset '{args.dataset}' unavailable for Verbal Uncertainty.")

def _build_answer_prompt(question: str, context: str, args):
    if args.dataset.upper() == "GSM8K":
        return (
            "You are a helpful math problem solver. "
            "Think step by step to solve the following problem. "
            "Provide the numeric answer in the end of response.\n"
            f"Problem: {question}\n\n"
        )
    elif args.dataset.upper() == "SQUAD":
        return (
            "Read the following text and answer the question. "
            "If the answer is not in the text, reply with 'unanswerable'.\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        )
        
    raise NotImplementedError(f"Dataset '{args.dataset}' unavailable for Generation.")
    

def get_model_answer(
    args,
    model,
    tokenizer,
    batch,
    source_list,
    context_list,
    device,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    num_return_sequences: int = 1,
    top_p: float = 1.0,
):
    idx_tensor = batch["index"]
    idx_list = idx_tensor.tolist() if torch.is_tensor(idx_tensor) else list(idx_tensor)

    batch_texts = []
    batch_contexts = []
    
    for idx in idx_list:
        batch_texts.append(source_list[idx])
        if context_list is not None:
            batch_contexts.append(context_list[idx])
        else:
            batch_contexts.append(None)

    if hasattr(args, 'uncertainty') and args.uncertainty == 'Verbal':
        prompts = [_build_confidence_prompt(q, c, args) for q, c in zip(batch_texts, batch_contexts)]
    else:
        prompts = [_build_answer_prompt(q, c, args) for q, c in zip(batch_texts, batch_contexts)]

    # Tokenize
    enc = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_only = outputs[:, input_ids.size(1):]
    gen_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    return idx_list, batch_texts, gen_texts


def generate_and_get_probabilities(
    args,
    models,
    unlabeled_loader,
    max_new_tokens: int = 128,
):    
    model = models["backbone"]
    tokenizer = models.get("tokenizer")
    tokenizer.padding_side = 'left'
    assert tokenizer is not None, "tokenizer missing"

    collator = CustomCollatorWithStrings(tokenizer=tokenizer)
    unlabeled_loader.collate_fn = collator

    model.eval()
    device = model.get_input_embeddings().weight.device

    base_ds = _unwrap_base_dataset(unlabeled_loader.dataset)

    # --- Fetch Questions & Contexts ---
    if hasattr(base_ds, "data_texts"):
        source_list = base_ds.data_texts
    elif hasattr(base_ds, "questions"):
        source_list = base_ds.questions
    else:
        raise ValueError("Dataset must have either 'data_texts' or 'questions'.")
    
    context_list = None
    if hasattr(base_ds, "contexts"):
        context_list = base_ds.contexts
    # ----------------------------------

    all_probabilities = []
    total_count = 0
    parsed_count = 0

    with torch.inference_mode():
        for batch in tqdm(unlabeled_loader, desc="Verbal Uncertainty", unit="batch"):
            idx_list, batch_texts, gen_texts = get_model_answer(
                args, model, tokenizer, batch, source_list, context_list, device,
                max_new_tokens=max_new_tokens
            )

            for raw_output in gen_texts:
                total_count += 1
                text_lower = raw_output.lower()
                
                pattern = r'(?:probability|confidence)[:\s]+(\d+(?:\.\d+)?)'
                match = re.search(pattern, text_lower)

                parsed_prob = None
                if match:
                    try:
                        val = float(match.group(1))
                        if 0 <= val <= 1:
                            parsed_prob = val
                            parsed_count += 1
                    except:
                        pass

                if parsed_prob is None:
                    parsed_prob = 0.0
                
                all_probabilities.append(parsed_prob)

    success_rate = parsed_count / total_count if total_count > 0 else 0
    print(f"[Verbal] Total: {total_count}, Parsed: {parsed_count}, Rate: {success_rate:.2%}")

    return np.array(all_probabilities)

def get_consistent_score(args, models, dataloader):
    model = models["backbone"]
    tokenizer = models.get("tokenizer")
    tokenizer.padding_side = 'left'
    assert tokenizer is not None, "tokenizer missing"

    collator = CustomCollatorWithStrings(tokenizer=tokenizer)
    dataloader.collate_fn = collator

    model.eval()
    device = model.get_input_embeddings().weight.device

    base_ds = _unwrap_base_dataset(dataloader.dataset)

    if args.dataset == "SQUAD":
        max_new_tokens = 16
    else:
        max_new_tokens = 256

    # --- Fetch Questions & Contexts ---
    if hasattr(base_ds, "data_texts"):
        source_list = base_ds.data_texts
    elif hasattr(base_ds, "questions"):
        source_list = base_ds.questions
    else:
        raise ValueError("Dataset must have either 'data_texts' or 'questions'.")
        
    context_list = None
    if hasattr(base_ds, "contexts"):
        context_list = base_ds.contexts

    all_scores = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Self-Consistent", unit="batch"):
            idx_list, batch_texts, gen_texts = get_model_answer(
                args, model, tokenizer, batch, source_list, context_list, device, 
                max_new_tokens=max_new_tokens,
                num_return_sequences=args.consistent_k,
                do_sample=True,
                temperature=0.7
            )
            
            k = args.consistent_k
            batch_size = len(idx_list)
            
            for i in range(batch_size):
                start_pos = i * k
                end_pos = (i + 1) * k
                sample_generations = gen_texts[start_pos:end_pos]
                
                extracted_answers = []
                for raw_output in sample_generations:
                    if args.dataset == "GSM8K":
                        ans = _extract_last_number_from_text(raw_output.lower())
                        if ans is None: ans = "INVALID"
                    elif args.dataset == "SQUAD":
                        clean_out = raw_output.replace("Answer:", "").strip()
                        ans = _normalize_text(clean_out)
                        if not ans: ans = "INVALID"
                    else:
                        ans = raw_output.strip() # Fallback

                    extracted_answers.append(ans)
                
                valid_answers = [x for x in extracted_answers if x != "INVALID"]
                
                if not valid_answers:
                    score = 0.0
                else:
                    counts = Counter(valid_answers)
                    top_count = counts.most_common(1)[0][1]
                    score = top_count / len(extracted_answers)
                
                all_scores.append(score)

    return np.array(all_scores)

def get_perplexity_score(args, models, dataloader):
    model = models["backbone"]
    tokenizer = models.get("tokenizer")
    
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
            
    model.eval()
    device = model.get_input_embeddings().weight.device

    collator = CustomCollatorWithStrings(
        tokenizer=tokenizer
    )
    dataloader.collate_fn = collator

    all_scores = []
    
    print(f"Calculating Perplexity (Confidence) on {len(dataloader.dataset)} samples...")

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="PPL Generation", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,      
                return_dict_in_generate=True, 
                output_scores=True,           
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            for i in range(len(input_ids)):
                log_probs = transition_scores[i]
                
                valid_log_probs = log_probs[~torch.isinf(log_probs)]
                
                if len(valid_log_probs) == 0:
                    score = 0.0 
                else:
                    mean_log_prob = valid_log_probs.mean().item()
                    score = np.exp(mean_log_prob)
                                
                all_scores.append(score)

    return np.array(all_scores)
