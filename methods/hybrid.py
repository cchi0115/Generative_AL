from .almethod import ALMethod
from .uncertainty import Uncertainty
from .diversity import Diversity
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
        n_query = self.args.n_query
        
        n_candidates = min(len(scores), n_query * self.hybrid_beta)
        print(f"[Hybrid] Uncertainty Prior: Filtering top {n_candidates} uncertain samples...")
        
        candidate_rel_indices = np.argsort(scores)[:n_candidates]
        candidate_U_index = [self.U_index[i] for i in candidate_rel_indices]
        
        print(f"[Hybrid] Handing over to Diversity ({self.diversity_method}) for final selection...")
        div_strategy = Diversity(
            self.args, self.models,
            self.unlabeled_dst,
            candidate_U_index,
            diversity_method=self.diversity_method
        )
        
        selected_local_indices, _ = div_strategy.run()
        
        final_selected_indices = [candidate_rel_indices[i] for i in selected_local_indices]
        
        return final_selected_indices, scores[final_selected_indices]

    def _run_diversity_prior(self, scores):
        n_query = self.args.n_query
        print(f"[Hybrid] Diversity Prior: Extracting features for all samples...")

        div_strategy = Diversity(
            self.args, self.models, self.unlabeled_dst, self.U_index,
            diversity_method=self.diversity_method
        )

        if self.diversity_method == "LLMLabel":
            features = div_strategy.get_llm_tag_vectors(self.unlabeled_set)
        else:
            features = div_strategy.get_semantic_embeddings(self.unlabeled_set)

        print(f"[Hybrid] Clustering features into {n_query} groups...")
        kmeans = KMeans(n_clusters=n_query, random_state=self.args.seed, n_init=10).fit(features)
        cluster_labels = kmeans.labels_

        final_selected_indices = []
        final_scores = []
        
        for i in range(n_query):
            cluster_members = np.where(cluster_labels == i)[0]
            
            if len(cluster_members) == 0:
                continue

            member_scores = scores[cluster_members]
            
            best_local_idx = np.argmin(member_scores)
            best_global_idx = cluster_members[best_local_idx]
            
            final_selected_indices.append(best_global_idx)
            final_scores.append(scores[best_global_idx])

        return final_selected_indices, np.array(final_scores)