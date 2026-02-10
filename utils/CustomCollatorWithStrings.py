from transformers import DataCollatorForSeq2Seq

class CustomCollatorWithStrings:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.hf_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            padding=True,
        )

    def __call__(self, features):
        tensor_keys = {"input_ids", "attention_mask", "labels", "decoder_input_ids", "token_type_ids"}
        
        tensor_features = []
        metadata_batch = {}

        for item in features:
            t_item = {}
            for key, value in item.items():
                if key in tensor_keys:
                    t_item[key] = value
                else:
                    if key not in metadata_batch:
                        metadata_batch[key] = []
                    metadata_batch[key].append(value)
            
            tensor_features.append(t_item)
            
        batch = self.hf_collator(tensor_features)
        
        for key, values in metadata_batch.items():
            batch[key] = values
                
        return batch