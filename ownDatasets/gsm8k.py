import torch
from torch.utils.data import Dataset

class GSM8KCausalLMDataset(Dataset):
    def __init__(
        self, 
        hf_dataset, 
        tokenizer, 
        max_length: int = 256,
        use_cot: bool = True,
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cot = use_cot

        self.questions = list(hf_dataset["question"])
        self.answers   = list(hf_dataset["answer"])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.questions)

    @staticmethod
    def _extract_final_answer_from_gsm8k(answer_text: str) -> str:
        if "####" in answer_text:
            final = answer_text.split("####")[-1].strip()
            return final
        elif "The final answer is:" in answer_text:
            final = answer_text.split("The final answer is:")[-1].strip()
            return final
        # fallback: 沒有 '####' 就直接用原文
        return answer_text.strip()

    def __getitem__(self, idx):
        question = self.questions[idx]
        full_answer = self.answers[idx].replace("####", "The final answer is:")

        if self.use_cot:
            answer_text = full_answer.strip()
        else:
            final_ans = self._extract_final_answer_from_gsm8k(full_answer)
            answer_text = final_ans

        if self.use_cot:
            prompt = (
                "You are a helpful math problem solver. "
                "Read the following problem and solve it step by step. "
                "Finish with the final numeric answer.\n"
                f"Problem: {question}\n"
                "Answer: "
            )
        else:
            prompt = (
                "Read the following math word problem and answer with only the final numeric result. "
                "No explanation.\n"
                f"Problem: {question}\n"
                "Answer: "
            )

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        ans_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        max_prompt_len = max(1, self.max_length - int(ans_ids.size(0)))
        prompt_ids = prompt_ids[:max_prompt_len]

        # concat prompt + answer
        full_ids = torch.cat([prompt_ids, ans_ids], dim=0)
        full_ids = full_ids[:self.max_length]

        # padding
        pad_len = self.max_length - full_ids.size(0)
        pad_id = self.tokenizer.pad_token_id

        if pad_len > 0:
            padding = torch.full((pad_len,), pad_id, dtype=torch.long)
            full_ids = torch.cat([padding, full_ids], dim=0)

        attention_mask = (full_ids != pad_id).long()

        labels = full_ids.clone()
        labels[:prompt_ids.size(0) + pad_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": question,
            "answer_text": full_answer,
            "target_final_answer": self._extract_final_answer_from_gsm8k(full_answer),
            "index": idx,
        }
