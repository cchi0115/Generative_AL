from .almethod import ALMethod
from .uncertainty import Uncertainty
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from utils.CustomCollatorWithStrings import CustomCollatorWithStrings

class Hybrid(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, 
                 selection_method="Verbal", 
                 diversity_method="semantic_embedding", 
                 hybrid_strategy="uncertainty_prior", 
                 hybrid_beta=5, 
                 **kwargs):
        """
        Args:
            selection_method: "Verbal", "Self-Consistent", "Perplexity"
            diversity_method: "semantic_embedding", "LLMLabel"
            hybrid_strategy: 
                - "uncertainty_prior": Select top (beta * k) uncertain samples -> Cluster into k.
                - "diversity_prior": Cluster all data into k groups -> Pick most uncertain in each group.
            hybrid_beta: Ratio for pre-filtering in uncertainty_prior mode (default 10).
        """
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

        self.uncertainty_method = selection_method
        self.diversity_method = diversity_method
        self.hybrid_strategy = hybrid_strategy
        self.hybrid_beta = hybrid_beta

        # Validate methods
        u_choices = ["Verbal", "Self-Consistent", "Perplexity"]
        d_choices = ["semantic_embedding", "LLMLabel"]
        
        if selection_method not in u_choices:
            raise NotImplementedError(f"Uncertainty method '{selection_method}' unavailable.")
        if diversity_method not in d_choices:
            raise NotImplementedError(f"Diversity method '{diversity_method}' unavailable.")

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index, scores

    def run(self):
        print(f"[Hybrid] Calculating Uncertainty Scores using {self.uncertainty_method}...")
        base_ds = self.unlabeled_set.dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
            
        max_u_idx = max(self.U_index)
        ds_len = len(base_ds)
        
        print(f"[Debug] Dataset Length: {ds_len}")
        print(f"[Debug] Max U_index: {max_u_idx}")
        
        if max_u_idx >= ds_len:
            raise ValueError(f"CRITICAL: U_index contains {max_u_idx}, but dataset size is only {ds_len}. Index mismatch!")
        
        unc_strategy = Uncertainty(
            self.args, self.models, base_ds, self.U_index, 
            selection_method=self.uncertainty_method
        )

        uncertainty_scores = unc_strategy.rank_uncertainty()
        
        if self.hybrid_strategy == "uncertainty_prior":
            return self._run_uncertainty_prior(uncertainty_scores)
        elif self.hybrid_strategy == "diversity_prior":
            return self._run_diversity_prior(uncertainty_scores)
        else:
            raise ValueError(f"Unknown hybrid strategy: {self.hybrid_strategy}")

    def _run_uncertainty_prior(self, scores):
        """
        Strategy: Filter -> Cluster
        1. Pick top (N * beta) most uncertain samples.
        2. Compute embeddings ONLY for these candidates.
        3. Cluster them into N groups and pick centers.
        """
        n_query = self.args.n_query
        n_candidates = min(len(scores), n_query * self.hybrid_beta)
        
        print(f"[Hybrid] Uncertainty Prior: Pre-selecting top {n_candidates} samples...")
        
        candidate_rel_indices = np.argsort(scores)[:n_candidates]
        candidate_subset = torch.utils.data.Subset(self.unlabeled_set, candidate_rel_indices)
        
        embeddings = self._get_embeddings(candidate_subset)
        
        print(f"[Hybrid] Clustering {len(embeddings)} candidates into {n_query} samples...")
        kmeans = KMeans(n_clusters=n_query, random_state=self.args.seed, n_init=10, init='k-means++')
        kmeans.fit(embeddings)
        
        dists = kmeans.transform(embeddings)
        min_indices_in_candidate = np.argmin(dists, axis=0)
        
        final_selected_indices = [candidate_rel_indices[i] for i in min_indices_in_candidate]
        
        final_scores = scores[final_selected_indices]
        
        return final_selected_indices, final_scores

    def _run_diversity_prior(self, scores):
        """
        Strategy: Cluster -> Filter
        1. Compute embeddings for ALL samples.
        2. Cluster all samples into N groups.
        3. In each group, pick the sample with the LOWEST uncertainty score.
        """
        n_query = self.args.n_query
        print(f"[Hybrid] Diversity Prior: Computing embeddings for all {len(self.unlabeled_set)} samples...")
        
        embeddings = self._get_embeddings(self.unlabeled_set)
        
        # K-Means Clustering
        print(f"[Hybrid] Clustering all samples into {n_query} groups...")
        kmeans = KMeans(n_clusters=n_query, random_state=self.args.seed, n_init=10, init='k-means++')
        cluster_labels = kmeans.fit_predict(embeddings)
        
        final_selected_indices = []
        final_scores = []
        
        for i in range(n_query):
            cluster_member_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_member_indices) == 0:
                continue
                
            member_scores = scores[cluster_member_indices]
            
            best_member_local_idx = np.argmin(member_scores)
            best_global_idx = cluster_member_indices[best_member_local_idx]
            
            final_selected_indices.append(best_global_idx)
            final_scores.append(scores[best_global_idx])
            
        return final_selected_indices, np.array(final_scores)

    def _get_embeddings(self, dataset):
        """
        Unified embedding retrieval method.
        Handles both 'semantic_embedding' and 'LLMLabel'.
        """
        tokenizer = self.models.get("tokenizer")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' # for collator

        collator = CustomCollatorWithStrings(tokenizer=tokenizer)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers,
            collate_fn=collator,
            shuffle=False
        )
        
        device = self.args.device

        text_to_embed = []
        
        if self.diversity_method == "LLMLabel":
            text_to_embed = self._generate_llm_labels(dataloader)
        else:
            raw_dataset = self.unlabeled_set.dataset if hasattr(self.unlabeled_set, "dataset") else self.unlabeled_set
            
            while hasattr(raw_dataset, "dataset"):
                raw_dataset = raw_dataset.dataset
            
            source_list = raw_dataset.questions if hasattr(raw_dataset, "questions") else raw_dataset.data_texts
            
            for batch in dataloader:
                indices = batch["index"]
                idx_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
                for idx in idx_list:
                    text_to_embed.append(source_list[idx])

        # --- Embedding ---
        print(f"[Hybrid] Embedding text using {self.args.diversity_embed_model}...")
        embed_tokenizer = AutoTokenizer.from_pretrained(self.args.diversity_embed_model)
        if embed_tokenizer.padding_side != 'right':
            embed_tokenizer.padding_side = 'right'
        embed_model = AutoModel.from_pretrained(self.args.diversity_embed_model).to(device)
        embed_model.eval()
        
        embeddings_list = []
        
        with torch.no_grad():
            batch_size = self.args.test_batch_size
            for i in tqdm(range(0, len(text_to_embed), batch_size), desc="Embedding"):
                batch_texts = text_to_embed[i : i + batch_size]
                
                encoded_input = embed_tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(device)
                
                outputs = embed_model(**encoded_input)
                
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                
                embeddings_list.append(sentence_embeddings.cpu().numpy())
                
        del embed_model
        torch.cuda.empty_cache()
        
        return np.concatenate(embeddings_list, axis=0)

    def _generate_llm_labels(self, dataloader):
        print(f"[Hybrid] Generating labels using {self.args.diversity_label_model}...")
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
        
        def _build_label_prompt(question):
            if self.args.dataset == 'GSM8K':
                return (
                    "Identify the core mathematical concepts and operations required to solve the following problem. "
                    "Output only 3-5 keywords separated by commas (e.g., Multiplication, Geometry, Ratio).\n\n"
                    f"Problem: {question}\n\n"
                    "Keywords:"
                )
            return "f{question}"
            
        generated_labels = []
        
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Labeling"):
                if "question" in batch:
                    questions = batch["question"]
                else:
                    # Fallback if collator structure changes
                    raise ValueError("Dataset must return 'question' field for LLMLabel.")
                
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
                generated_labels.extend(gen_text)
                
        del label_model
        torch.cuda.empty_cache()
        
        return generated_labels
