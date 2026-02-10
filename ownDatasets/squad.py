import torch
from torch.utils.data import Dataset

class SQuADV2CausalLMDataset(Dataset):
    def __init__(
        self, 
        hf_dataset, 
        tokenizer,
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer

        self.contexts = list(hf_dataset["context"])
        self.questions = list(hf_dataset["question"])
        self.answers = list(hf_dataset["answers"])
        self.ids = list(hf_dataset["id"])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer_obj = self.answers[idx] 

        # Get Answer Text
        if len(answer_obj["text"]) > 0:
            target_text = answer_obj["text"][0]
        else:
            target_text = "unanswerable"

        # Prompt (Instruction + Context + Question)
        prompt = (
            "Read the following text and answer the question. "
            "If the answer is not in the text, reply with 'unanswerable'.\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        )

        # Tokenize Prompt
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        # Tokenize Answer
        ans_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        # Concat Prompt + Answer
        full_ids = torch.cat([prompt_ids, ans_ids], dim=0)
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)

        labels = full_ids.clone()
        labels[:prompt_ids.size(0)] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": question,
            "context": context, 
            "answers": answer_obj,    
            "target_final_answer": target_text,
            "id": self.ids[idx],
            "index": idx,
        }
