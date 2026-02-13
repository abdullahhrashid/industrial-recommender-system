import torch
import numpy as np

#a helper class for evaluation
class RetrievalMetrics:
    def __init__(self, k_values=[10, 20, 50, 100]):
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        self.recalls = {k: [] for k in self.k_values}
        self.ndcgs = {k: [] for k in self.k_values}
        self.mrrs = []
    
    @torch.no_grad()
    def update(self, scores, target_indices):
        batch_size = scores.size(0)
        
        target_scores = scores[torch.arange(batch_size), target_indices].unsqueeze(1)
        
        ranks = (scores > target_scores).sum(dim=1) + 1
        
        ranks = ranks.cpu().numpy()
        
        for rank in ranks:
            self.mrrs.append(1.0 / rank)
    
            for k in self.k_values:
                if rank <= k:
                    self.recalls[k].append(1.0)
                    self.ndcgs[k].append(1.0 / np.log2(rank + 1))
                else:
                    self.recalls[k].append(0.0)
                    self.ndcgs[k].append(0.0)
    
    def compute(self):
        results = {} 
        results['MRR'] = np.mean(self.mrrs) if self.mrrs else 0.0
        for k in self.k_values:
            results[f'Recall@{k}'] = np.mean(self.recalls[k]) if self.recalls[k] else 0.0
            results[f'NDCG@{k}'] = np.mean(self.ndcgs[k]) if self.ndcgs[k] else 0.0
        return results
    