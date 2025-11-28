import os
import csv
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

# 給 Llama 用的 prompt 模板
PROMPT_HEADER_CASUALLM = (
    "You are a multiple-choice classifier. "
    "Read the news and answer in JSON with exactly these keys, no extra text: "
    '{"answer": A/B/C/D, "confidence": Very unconfident/Unconfident/Neutral/Confident/Very confident}.\n'
    "Answer options: A. World, B. Sports, C. Business, D. Sci/Tech\n\n"
)

JSON_PATTERN_ANSWER_CONF = re.compile(
    r'{?\s*"answer"\s*:\s*"([ABCD])"\s*,\s*"confidence"\s*:\s*(\d{1,3})\s*}?',
    re.IGNORECASE | re.DOTALL
)

def _build_prompt_casuallm_no_confidence(text: str) -> str:
    return (
        "You are a multiple-choice classifier. "
        + "Read the news and answer in a single character 'A', 'B', 'C' or 'D', no extra text.\n"
        + "Answer Options:\nA. World\nB. Sports\nC. Business\nD. Sci/Tech\n\n"
        + f"News: {text}/n"
        + f"Answer: "
    )

def _build_prompt_casuallm(text: str) -> str:
    return (
        PROMPT_HEADER_CASUALLM
        + f"News: {text}/n"
        + f"JSON: "
    )

def _unwrap_base_dataset(ds):
    """
    如果 dataloader.dataset 是 Subset，就一直往內 unwrap，
    最後拿到真正的 base dataset（例如 AGNewsCausalLMOptionDataset）。
    """
    base = ds
    while isinstance(base, Subset):
        base = base.dataset
    return base

def generate_unlabeled_casuallm_with_confidence(
    args,
    models,
    unlabeled_loader,
    output_path,
    max_source_length: int = 768,
    max_new_tokens: int = 20,
    gen_batch_size: int = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    """
    使用 causal LM (Llama) 對 unlabeled_loader 中的所有樣本做 prompt 生成，
    要求模型輸出 JSON:
        {"answer": "A|B|C|D", "confidence": 0-100}

    並將結果寫入 CSV：
        index, answer, confidence, raw_output, prompt_preview

    - index 來自 batch['index']，可以跟 AL 的 index 保持一致
    - answer / confidence 完全來自模型生成的文字（self-reported）
    """

    model = models["backbone"]
    tokenizer = models.get("tokenizer", None)
    assert tokenizer is not None, "tokenizer is required for causal LM generation."

    model.eval()

    # 用 embedding 權重所在裝置，避免和 device_map 打架
    device = model.get_input_embeddings().weight.device

    # 確保有 pad_token（Llama 常沒有）
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 取 base dataset（假設裡面有 data_texts）
    base_ds = _unwrap_base_dataset(unlabeled_loader.dataset)
    # 你前面在 AGNewsCausalLMOptionDataset 有 self.data_texts
    data_texts = getattr(base_ds, "data_texts", None)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fout = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    writer.writerow(["index", "answer", "confidence", "raw_output", "prompt_preview"])

    with torch.inference_mode():
        for batch in tqdm(unlabeled_loader, desc="Generating", unit="batch"):
            # 這裡假設你的 dataset 有回 "index" 欄位（AGNewsCausalLMOptionDataset 是有的）
            idx_tensor = batch["index"]  # [B]
            if torch.is_tensor(idx_tensor):
                idx_list = idx_tensor.tolist()
            else:
                idx_list = list(idx_tensor)

            # 從 base_ds.data_texts 取得原始 news text
            batch_texts = []
            for idx in idx_list:
                if data_texts is not None:
                    batch_texts.append(data_texts[idx])
                else:
                    # fallback: 若沒有 data_texts，就 decode 輸入（但可能包含 prompt）
                    input_ids_i = batch["input_ids"][idx_list.index(idx)]
                    text_i = tokenizer.decode(
                        input_ids_i.tolist() if torch.is_tensor(input_ids_i) else input_ids_i,
                        skip_special_tokens=True,
                    )
                    batch_texts.append(text_i)

            # 建 prompt 列表
            prompts = [_build_prompt_casuallm(t) for t in batch_texts]

            # tokenize prompts
            enc = tokenizer(
                prompts,
                padding=True,
                # truncation=True,
                # max_length=max_source_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)

            # 生成參數
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Llama 生成
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # 只 decode 新生成部分（去掉 prompt）
            gen_only = outputs[:, input_ids.size(1):]
            gen_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            # 逐筆解析 answer / confidence
            for idx_val, raw, prompt in zip(idx_list, gen_texts, prompts):
                m = JSON_PATTERN_ANSWER_CONF.search(raw)
                if m:
                    ans = m.group(1).upper()
                    conf = int(m.group(2))
                    conf = max(0, min(100, conf))  # clamp 到 0~100
                else:
                    ans, conf = "?", 0  # 解析失敗 → 當作未知

                writer.writerow([
                    idx_val,
                    ans,
                    conf,
                    raw.strip().replace("\n", "\\n"),
                    prompt.replace("\n", " "),
                ])

    del input_ids, attention_mask, outputs, gen_only, gen_texts
    torch.cuda.empty_cache()
    
    fout.close()
    print(f"[Generative AL] Saved unlabeled predictions to: {output_path}")
    return output_path
