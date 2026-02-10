from .almethod import ALMethod
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.generate_result import _build_answer_prompt
from utils.CustomCollatorWithStrings import CustomCollatorWithStrings
from collections import Counter

class Diversity(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, diversity_method="semantic_embedding", **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

        selection_choices = [
            "semantic_embedding", "LLMLabel"
        ]
        if diversity_method not in selection_choices:
            raise NotImplementedError(f"Selection algorithm '{diversity_method}' unavailable.")
        
        self.selection_method = diversity_method
  
    def select(self, **kwargs):
        """
        Exposed method: Returns selected unlabeled sample indices and their scores.
        """
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def run(self, **kwargs):
        model = self.models.get("backbone")
        tokenizer = self.models.get("tokenizer")
        tokenizer.padding_side = 'left'
        device = self.args.device
        
        model.eval()

        collator = CustomCollatorWithStrings(
            tokenizer=tokenizer
        )
        
        selection_loader = torch.utils.data.DataLoader(
            self.unlabeled_set, 
            batch_size=self.args.test_batch_size, 
            num_workers=self.args.workers,
            collate_fn=collator,
        )        

        if self.selection_method == "semantic_embedding":
            print(f"[Diversity] Loading Embedding Model: {self.args.diversity_embed_model}...")
            embed_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_embed_model)
            embed_model = AutoModel.from_pretrained(self.args.diversity_embed_model).to(device)
            embed_model.eval()

            selected_indices, scores = self.get_semantic_embedding_score(embed_model, embed_tokenizer, selection_loader, device)
            return selected_indices, scores
        elif self.selection_method == "LLMLabel":
            del self.models

            print(f"[Diversity] Loading Labeling Model: {self.args.diversity_label_model}...")
            label_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_label_model)
            label_tokenizer.padding_side = 'left'
            if label_tokenizer.pad_token is None:
                label_tokenizer.pad_token = label_tokenizer.eos_token

            label_model = AutoModelForCausalLM.from_pretrained(
                self.args.diversity_label_model,
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            label_model.eval()

            selected_indices, scores = self.get_llm_label_score(label_model, label_tokenizer, selection_loader, device)
            return selected_indices, scores

    def get_semantic_embedding_score(self, embed_model, embed_tokenizer, dataloader, device):
        embeddings_list = []        

        print("[Diversity] Computing Embeddings for CoT traces...")
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc="Embedding"):
                batch_texts = batch['question']
                
                encoded_input = embed_tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(device)
                
                outputs = embed_model(**encoded_input)
                
                # --- Mean Pooling ---
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                sentence_embeddings = sum_embeddings / sum_mask
                
                # --- L2 Normalization ---
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                
                embeddings_list.append(sentence_embeddings.cpu().numpy())
                
        del embed_model
        torch.cuda.empty_cache()
        
        all_embeddings = np.concatenate(embeddings_list, axis=0)
        
        n_query = self.args.n_query
        print(f"[Diversity] Running K-Means clustering (K={n_query})...")
        
        kmeans = KMeans(n_clusters=n_query, random_state=self.args.seed, n_init=10)
        kmeans.fit(all_embeddings)
        
        dists = kmeans.transform(all_embeddings)
        
        selected_local_indices = []
        min_indices = np.argmin(dists, axis=0) 
        
        selected_local_indices = min_indices.tolist()

        scores = dists[min_indices, np.arange(n_query)]
        # scores = list(np.ones(len(selected_local_indices)))
            
        return selected_local_indices, scores
    
    def get_llm_label_score(self, label_model, label_tokenizer, dataloader, device):
        TAG_SIMILARITY_THRESHOLD = 0.2  
        MIN_TAG_FREQUENCY = 5    
        MAX_TAG_WORDS = 4        

        def _build_label_prompt(question):
            return (
                "Identify the core mathematical concepts and operations required to solve the following problem. "
                "Output keywords separated by commas (e.g., Multiplication, Geometry, Ratio).\n\n"
                f"Problem: {question}\n\n"
                "Keywords:"
            )

        # --- Phase 1: Generate Labels ---
        raw_generated_texts = [] 

        print("[Diversity] Phase 1: Generating Labels with LLM...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Labeling"):
                questions = batch["question"]
                prompts = [_build_label_prompt(q) for q in questions]
                
                inputs = label_tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(label_model.device)
                
                outputs = label_model.generate(
                    **inputs,
                    max_new_tokens=32, 
                    do_sample=False,
                    pad_token_id=label_tokenizer.pad_token_id
                )
                
                gen_text = label_tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                raw_generated_texts.extend(gen_text)

        print("[Diversity] Freeing Label Model memory...")
        del label_model, label_tokenizer
        torch.cuda.empty_cache()

        # --- Phase 2: Parse & Clean Tags ---
        # 1. Split & Normalize
        all_sample_tags = []
        unique_tags_set = set()

        print("[Diversity] Parsing and cleaning tags...")
        for text in raw_generated_texts:
            tags = text.split(',')
            valid_tags = []
            for t in tags:
                t = t.strip().lower()
                if t and len(t.split()) < MAX_TAG_WORDS:
                    valid_tags.append(t)
                    unique_tags_set.add(t)
            all_sample_tags.append(valid_tags)
        
        unique_tags_list = list(unique_tags_set)
        print(f"[Diversity] Found {len(unique_tags_list)} unique raw tags.")

        # --- Phase 3: Embed Tags & Semantic Merge ---
        print(f"[Diversity] Phase 3: Embedding {len(unique_tags_list)} Unique Tags for merging...")
        
        embed_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_embed_model)
        if embed_tokenizer.padding_side != 'right':
            embed_tokenizer.padding_side = 'right'
        embed_model = AutoModel.from_pretrained(self.args.diversity_embed_model).to(device)
        embed_model.eval()

        tag_embeddings = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(unique_tags_list), batch_size):
                batch_tags = unique_tags_list[i : i + batch_size]
                encoded_input = embed_tokenizer(batch_tags, padding=True, truncation=True, return_tensors='pt').to(device)
                outputs = embed_model(**encoded_input)
                
                # Pooling
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                emb = F.normalize(sum_embeddings / sum_mask, p=2, dim=1)
                
                tag_embeddings.append(emb.cpu().numpy())
        
        del embed_model
        torch.cuda.empty_cache()
        
        tag_embeddings_matrix = np.concatenate(tag_embeddings, axis=0) # (N_tags, Dim)

        print(f"[Diversity] Merging semantically similar tags (Threshold={TAG_SIMILARITY_THRESHOLD})...")
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=TAG_SIMILARITY_THRESHOLD,
            metric='euclidean', 
            linkage='average'
        )
        clustering.fit(tag_embeddings_matrix)
        
        tag_to_concept = {tag: label for tag, label in zip(unique_tags_list, clustering.labels_)}
        n_concepts = clustering.n_clusters_
        print(f"[Diversity] Merged {len(unique_tags_list)} tags into {n_concepts} semantic concepts.")

        # --- Phase 4: Filter Long-Tail & Vectorize Documents ---
        concept_counter = Counter()
        for tags in all_sample_tags:
            concepts = set(tag_to_concept[t] for t in tags if t in tag_to_concept)
            concept_counter.update(concepts)
        
        valid_concepts = {c for c, count in concept_counter.items() if count >= MIN_TAG_FREQUENCY}
        print(f"[Diversity] Kept {len(valid_concepts)} concepts after filtering rare ones (Freq < {MIN_TAG_FREQUENCY}).")
        
        sorted_valid_concepts = sorted(list(valid_concepts))
        concept_to_col = {c: i for i, c in enumerate(sorted_valid_concepts)}
        n_features = len(sorted_valid_concepts)

        if n_features == 0:
            print("[Warning] No valid concepts found after filtering! Falling back to random selection.")
            return list(range(self.args.n_query)), list(np.zeros(self.args.n_query))

        
        doc_vectors_bool = np.zeros((len(raw_generated_texts), n_features), dtype=bool)
        
        for i, tags in enumerate(all_sample_tags):
            for t in tags:
                if t in tag_to_concept:
                    c_id = tag_to_concept[t]
                    if c_id in concept_to_col:
                        doc_vectors_bool[i, concept_to_col[c_id]] = True

        # --- Phase 5: K-Means on Concept Vectors ---
        n_query = self.args.n_query
        print(f"[Diversity] Phase 5: Greedy selection of {n_query} samples to maximize concept coverage...")
        
        selected_indices = []
        covered_mask = np.zeros(n_features, dtype=bool)
        is_selected = np.zeros(len(doc_vectors_bool), dtype=bool)
        
        sample_concept_counts = np.sum(doc_vectors_bool, axis=1)
        
        for _ in tqdm(range(n_query), desc="Greedy Selection"):
            uncovered_concepts = ~covered_mask
            
            if not np.any(uncovered_concepts):
                current_max_gain = 0
            else:
                gains = np.sum(doc_vectors_bool[:, uncovered_concepts], axis=1)
                gains[is_selected] = -1
                
                best_idx = np.argmax(gains)
                current_max_gain = gains[best_idx]
            
            if current_max_gain > 0:
                pass 
            else:
                remaining_indices = np.where(~is_selected)[0]
                
                if len(remaining_indices) == 0:
                    break
                
                best_relative_idx = np.argmax(sample_concept_counts[remaining_indices])
                best_idx = remaining_indices[best_relative_idx]
            
            selected_indices.append(best_idx)
            is_selected[best_idx] = True
            covered_mask |= doc_vectors_bool[best_idx]
            
        selected_scores = np.ones(len(selected_indices))
        
        return selected_indices, selected_scores
