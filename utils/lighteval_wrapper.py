import os
import gc
import torch
import shutil
import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoConfig

# LightEval imports
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.adapter_model import AdapterModel, AdapterModelConfig
# from lighteval.models.model_config import GenerationParameters

# =============================================================================
# [FIX] Monkey Patch: 修復 LightEval 0.10.0 AdapterModel 的嚴重缺陷
# 源碼問題：
# 1. 試圖存取不存在的 config.pretrained
# 2. Config 中 adapter_weights 型別被定義為 bool
# 3. 強制寫入硬碟導致效率低落
# =============================================================================
def fixed_create_auto_model(self):
    logger = logging.getLogger(__name__)
    
    # 1. 決定 Adapter 路徑
    # 由於 Config 定義混亂，我們約定 model_name 就是 adapter 的路徑
    # (因為 TransformersModelConfig 的 model_name 通常是主路徑)
    adapter_path = self.config.model_name
    
    logger.info(f"[MonkeyPatch] Loading base model: {self.config.base_model}")
    logger.info(f"[MonkeyPatch] Loading adapter from: {adapter_path}")

    # 2. 載入 Base Model
    # 使用 device_map="auto" 讓 accelerate 自動分配資源
    base_model = AutoModelForCausalLM.from_pretrained(
        self.config.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=self.config.trust_remote_code,
        token=True
    )

    # 3. 載入並掛載 Adapter (In-Memory, 不存檔)
    # 直接回傳 PeftModel，不需要 merge_and_unload 也不需要 save_pretrained
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        is_trainable=False 
    )

    print("="*20 + " [Debug] Model Architecture Loaded by LightEval " + "="*20)
    print(model)
    print("="*80)

    return model

# 應用 Patch：替換掉原本壞掉的函式
AdapterModel._create_auto_model = fixed_create_auto_model
# =============================================================================


def evaluate_with_lighteval(args, models, trial, cycle, prefix):
    """
    Wrapper for GSM8K evaluation with LightEval.
    Handles device_map='auto', tokenizer loading, and applies AdapterModel fix.
    """
    print(f"\n[LightEval] Starting GSM8K evaluation for Trial {trial+1} Cycle {cycle+1}...")

    # 1. Validation
    if not isinstance(models, dict) or 'backbone' not in models:
        print("[LightEval] Error: 'models' expected to be a dict with 'backbone'. Skipping.")
        return 0.0
    
    peft_model = models['backbone']
    save_dir = getattr(args, "save_dir", "./checkpoints")
    adapter_save_path = os.path.join(save_dir, f"{prefix}_trial{trial}_cycle{cycle}")
    
    # 2. Save Adapter and Tokenizer
    # LightEval 需要從磁碟讀取這些檔案
    print(f"[LightEval] Saving temporary adapter to {adapter_save_path}...")
    peft_model.save_pretrained(adapter_save_path)
    
    tokenizer_path = adapter_save_path
    if 'tokenizer' in models:
        print(f"[LightEval] Saving tokenizer to {adapter_save_path}...")
        models['tokenizer'].save_pretrained(adapter_save_path)
    else:
        # Fallback if no tokenizer is present (rare)
        tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"

    # 3. Nuke Model to Free VRAM
    # 這是為了避免 OOM，因為 LightEval 會建立新的模型實例
    print("[LightEval] NUKING training model to free memory for evaluation...")
    saved_tokenizer = models.get('tokenizer')
    del models['backbone'] 
    if 'tokenizer' in models:
        del models['tokenizer']
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()
    
    acc = 0.0
    try:
        base_model_name = "meta-llama/Llama-2-7b-chat-hf"


        # 設定適合 GSM8K 的參數
        # gen_params = GenerationParameters(
        #     temperature=0.0,        # 貪婪解碼 (Greedy Decoding)，最適合數學題
        #     do_sample=False,        # 關閉隨機取樣
        #     max_new_tokens=512,     # 給予足夠的空間進行推理 (Chain-of-Thought)
        #     top_p=1.0,              # 不進行 Nucleus 截斷
        # )

        # 4. Configure LightEval
        # [關鍵設定] 配合 Monkey Patch 的邏輯：
        model_config = AdapterModelConfig(
            # Patch 會讀取 model_name 作為 adapter_path
            model_name=adapter_save_path,      
            
            # 這是 Base Model
            base_model=base_model_name,        
            
            # 因為原始碼定義 adapter_weights: bool，我們傳 True 騙過 Pydantic 驗證
            adapter_weights=True,              
            
            # 指向包含 tokenizer.json 的本地路徑
            tokenizer=tokenizer_path,          
            
            dtype="float16", 
            trust_remote_code=True,
        #     generation_parameters=gen_params
        )

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE, 
            max_samples=None 
        )

        eval_output_dir = os.path.join("outputs", "lighteval_results")
        tracker = EvaluationTracker(output_dir=eval_output_dir, save_details=True)

        # 使用標準 GSM8K 任務字串 (suite|task|few_shot|truncate)
        # 您也可以換成自定義任務 "custom|utils/my_gsm8k.py|0|0"
        task_string = "lighteval|gsm8k|0|0"

        pipeline = Pipeline(
            tasks=task_string,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=tracker,
            model_config=model_config
        )

        pipeline.evaluate()
        pipeline.save_and_push_results()
        pipeline.show_results()

        results = pipeline.get_results()
        for task_key, metrics in results.get("results", {}).items():
            if "gsm8k" in task_key:
                # 嘗試抓取各種可能的 metric key
                acc = metrics.get("acc", metrics.get("extractive_match", 0.0))
                break
        # acc = results['results']['all']['extractive_match']

        print(f"[LightEval] Evaluation finished. GSM8K Accuracy: {acc}")

    except Exception as e:
        print(f"[LightEval] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 7. Reload Model & Restore Gradients
        print("[LightEval] Reloading model with device_map='auto'...")
        
        # [重要] 恢復梯度計算，否則 main.py 後續的 backward() 會失敗
        torch.set_grad_enabled(True) 
        
        try:
            base_config = AutoConfig.from_pretrained(base_model_name, token=True)
            new_base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=base_config,
                device_map="auto",
                trust_remote_code=True,
                token=True
            )
            
            # 重新載入 Adapter，設為可訓練
            new_peft_model = PeftModel.from_pretrained(
                new_base_model, 
                adapter_save_path, 
                is_trainable=True 
            )
            
            # 確保模型處於訓練模式且 LoRA 參數需要梯度
            new_peft_model.train() 
            for name, param in new_peft_model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

            models['backbone'] = new_peft_model
            if saved_tokenizer:
                models['tokenizer'] = saved_tokenizer
            
            print("[LightEval] Model reloaded successfully.")
            
        except Exception as e:
            print(f"[LightEval] CRITICAL ERROR: Failed to reload model! {e}")
            raise e

    return acc
