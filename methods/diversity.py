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
        self.TAG_SIMILARITY_THRESHOLD = 0.2
        self.MIN_TAG_FREQUENCY = 5
        self.MAX_TAG_WORDS = 4
  
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
        model.eval()
        
        n_query = self.args.n_query

        if self.selection_method == "LLMLabel":
            features = self.get_llm_tag_vectors(self.unlabeled_set)
            
            selected_indices = self.greedy_max_coverage(features, n_query)
            scores = np.ones(len(selected_indices))
            
        elif self.selection_method == "semantic_embedding":
            features = self.get_semantic_embeddings(self.unlabeled_set)
            
            print(f"[Diversity] Running K-Means clustering (K={n_query})...")
            kmeans = KMeans(n_clusters=n_query, random_state=self.args.seed, n_init=10, init='k-means++')
            kmeans.fit(features)
            
            dists = kmeans.transform(features)
            selected_indices = np.argmin(dists, axis=0).tolist()
            scores = dists[selected_indices, np.arange(n_query)]
            
        else:
            raise NotImplementedError(f"Diversity method '{self.selection_method}' unavailable.")

        return selected_indices, scores
    
    def get_llm_tag_vectors(self, dataset):
        # 1. Generate Labels
        raw_texts = self._generate_llm_labels(dataset)
        # 2. Parse, Merge & Vectorize
        return self._process_tags_to_vectors(raw_texts)

    def get_semantic_embeddings(self, dataset):
        return self._generate_embeddings_from_dataset(dataset)

    def greedy_max_coverage(self, doc_vectors_bool, n_query):
        print(f"[Diversity] Running Greedy Max Coverage selection for {n_query} samples...")
        
        selected_indices = []
        n_samples, n_features = doc_vectors_bool.shape
        
        covered_mask = np.zeros(n_features, dtype=bool)       
        is_selected = np.zeros(n_samples, dtype=bool)         
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
            
        return selected_indices

    def _generate_llm_labels(self, dataset):
        print(f"[Diversity] Generating labels using {self.args.diversity_label_model}...")
        
        label_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_label_model)
        label_tokenizer.padding_side = 'left'
        if label_tokenizer.pad_token is None: label_tokenizer.pad_token = label_tokenizer.eos_token

        label_model = AutoModelForCausalLM.from_pretrained(
            self.args.diversity_label_model, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        label_model.eval()
        
        def _build_label_prompt(question):
            return (
                "Identify the core mathematical concepts and operations required to solve the following problem. "
                "Output keywords separated by commas (e.g., Multiplication, Geometry, Ratio).\n\n"
                f"Problem: {question}\n\n"
                "Keywords:"
            )
            
        generated_labels = []
        batch_size = self.args.test_batch_size
        
        dataset_len = len(dataset)
        print(f"[Diversity] Processing {dataset_len} samples manually...")
        
        with torch.no_grad():
            for i in tqdm(range(0, dataset_len, batch_size), desc="Labeling"):
                current_indices = range(i, min(i + batch_size, dataset_len))
                
                prompts = []
                for idx in current_indices:
                    item = dataset[idx] 
                    
                    if "question" in item:
                        q = item["question"]
                    elif "text" in item:
                        q = item["text"]
                    else:
                        print(f"DEBUG: Item keys at index {idx}: {item.keys()}")
                        raise ValueError(f"Item at index {idx} missing 'question' field.")
                        
                    prompts.append(_build_label_prompt(q))

                # Tokenize
                inputs = label_tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(label_model.device)
                
                # Generate
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
                generated_labels.extend(gen_text)
                
        del label_model
        torch.cuda.empty_cache()
        print("generated labels:", len(generated_labels), generated_labels[0])
        return generated_labels

    def _process_tags_to_vectors(self, raw_texts):
        # 1. Parse Tags
        all_sample_tags = []
        unique_tags_set = set()
        
        print("[Diversity] Parsing and cleaning tags...")
        for text in raw_texts:
            tags = text.split(',')
            valid_tags = []
            for t in tags:
                t = t.strip().lower()
                if t and len(t.split()) < self.MAX_TAG_WORDS:
                    valid_tags.append(t)
                    unique_tags_set.add(t)
            all_sample_tags.append(valid_tags)
        
        unique_tags_list = list(unique_tags_set)
        print(f"[Diversity] Found {len(unique_tags_list)} unique raw tags.")
        
        # 2. Embed Unique Tags
        print(f"[Diversity] Embedding tags for merging...")
        embed_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_embed_model)
        if embed_tokenizer.padding_side != 'right': embed_tokenizer.padding_side = 'right'
        embed_model = AutoModel.from_pretrained(self.args.diversity_embed_model).to(self.args.device)
        embed_model.eval()
        
        tag_embeddings = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(unique_tags_list), batch_size):
                batch_tags = unique_tags_list[i : i+batch_size]
                encoded_input = embed_tokenizer(batch_tags, padding=True, truncation=True, return_tensors='pt').to(self.args.device)
                outputs = embed_model(**encoded_input)
                
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                emb = F.normalize(sum_embeddings / sum_mask, p=2, dim=1)
                tag_embeddings.append(emb.cpu().numpy())
        
        del embed_model
        torch.cuda.empty_cache()
        tag_embeddings_matrix = np.concatenate(tag_embeddings, axis=0)
        
        # 3. Merge Tags (Agglomerative Clustering)
        print(f"[Diversity] Merging similar tags (Threshold={self.TAG_SIMILARITY_THRESHOLD})...")
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.TAG_SIMILARITY_THRESHOLD,
            metric='euclidean', 
            linkage='average'
        )
        clustering.fit(tag_embeddings_matrix)
        tag_to_concept = {tag: label for tag, label in zip(unique_tags_list, clustering.labels_)}
        
        # 4. Filter Long-tail & Construct Matrix
        concept_counter = Counter()
        for tags in all_sample_tags:
            concepts = set(tag_to_concept[t] for t in tags if t in tag_to_concept)
            concept_counter.update(concepts)
            
        valid_concepts = {c for c, count in concept_counter.items() if count >= self.MIN_TAG_FREQUENCY}
        sorted_valid_concepts = sorted(list(valid_concepts))
        concept_to_col = {c: i for i, c in enumerate(sorted_valid_concepts)}
        n_features = len(sorted_valid_concepts)
        
        print(f"[Diversity] Final Concept Space: {n_features} dimensions.")
        
        doc_vectors_bool = np.zeros((len(raw_texts), n_features), dtype=bool)
        for i, tags in enumerate(all_sample_tags):
            for t in tags:
                if t in tag_to_concept:
                    c_id = tag_to_concept[t]
                    if c_id in concept_to_col:
                        doc_vectors_bool[i, concept_to_col[c_id]] = True
                        
        return doc_vectors_bool

    def _generate_embeddings_from_dataset(self, dataset):
        print(f"[Diversity] Embedding full text using {self.args.diversity_embed_model}...")
        
        tokenizer = self.models.get("tokenizer")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        collator = CustomCollatorWithStrings(tokenizer=tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.test_batch_size, num_workers=self.args.workers,
            collate_fn=collator, shuffle=False
        )

        # Extract text list
        raw_dataset = dataset
        while hasattr(raw_dataset, "dataset"): raw_dataset = raw_dataset.dataset
        source_list = raw_dataset.questions if hasattr(raw_dataset, "questions") else raw_dataset.data_texts
        
        text_to_embed = []
        for batch in dataloader:
            indices = batch["index"]
            idx_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
            for idx in idx_list:
                text_to_embed.append(source_list[idx])

        # Load Embed Model
        embed_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_embed_model)
        if embed_tokenizer.padding_side != 'right': embed_tokenizer.padding_side = 'right'
        embed_model = AutoModel.from_pretrained(self.args.diversity_embed_model).to(self.args.device)
        embed_model.eval()
        
        embeddings_list = []
        with torch.no_grad():
            batch_size = self.args.test_batch_size
            for i in tqdm(range(0, len(text_to_embed), batch_size), desc="Embedding"):
                batch_texts = text_to_embed[i : i + batch_size]
                
                encoded_input = embed_tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
                ).to(self.args.device)
                
                outputs = embed_model(**encoded_input)
                
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                emb = F.normalize(sum_embeddings / sum_mask, p=2, dim=1)
                
                embeddings_list.append(emb.cpu().numpy())
        
        del embed_model
        torch.cuda.empty_cache()
        return np.concatenate(embeddings_list, axis=0)
