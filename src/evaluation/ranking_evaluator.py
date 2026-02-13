import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss

class RankingMetrics:
    def __init__(self, ks=[5, 10, 20]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.all_scores = []
        self.all_labels = []

    def update(self, scores, labels):
        self.all_scores.append(scores.cpu())
        self.all_labels.append(labels.cpu())

    def ndcg_at_k(self, scores, labels, k):
        _, topk_indices = torch.topk(scores, k, dim=1)
        topk_labels = torch.gather(labels, 1, topk_indices)

        #dcg
        positions = torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(0)
        dcg = (topk_labels / torch.log2(positions + 1)).sum(dim=1)

        #ideal dcg
        ideal_sorted, _ = torch.sort(labels, dim=1, descending=True)
        ideal_topk = ideal_sorted[:, :k]
        idcg = (ideal_topk / torch.log2(positions + 1)).sum(dim=1)

        ndcg = dcg / (idcg + 1e-8)
        return ndcg.mean().item()

    def hit_rate_at_k(self, scores, labels, k):
        _, topk_indices = torch.topk(scores, k, dim=1)
        topk_labels = torch.gather(labels, 1, topk_indices)

        #hit if any positive in top k
        hits = (topk_labels.sum(dim=1) > 0).float()
        return hits.mean().item()

    def map_at_k(self, scores, labels, k):
        _, topk_indices = torch.topk(scores, k, dim=1)
        topk_labels = torch.gather(labels, 1, topk_indices)

        #cumulative sum of relevance
        cum_relevant = topk_labels.cumsum(dim=1)
        positions = torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(0)

        #precision at each position
        precision_at_pos = cum_relevant / positions

        #only counting positions where item is relevant
        ap = (precision_at_pos * topk_labels).sum(dim=1) / (labels.sum(dim=1) + 1e-8)
        return ap.mean().item()

    def compute(self):
        all_scores = torch.cat(self.all_scores, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)

        probs = torch.sigmoid(all_scores)

        metrics = {}

        #auc
        flat_probs = probs.reshape(-1).numpy()
        flat_labels = all_labels.reshape(-1).numpy()
        metrics['AUC'] = roc_auc_score(flat_labels, flat_probs)

        #log loss
        metrics['LogLoss'] = log_loss(flat_labels, np.clip(flat_probs, 1e-7, 1 - 1e-7))

        #ranking metrics at various k
        for k in self.ks:
            if all_scores.size(1) >= k:
                metrics[f'NDCG@{k}'] = self.ndcg_at_k(all_scores, all_labels, k)
                metrics[f'HitRate@{k}'] = self.hit_rate_at_k(all_scores, all_labels, k)
                metrics[f'MAP@{k}'] = self.map_at_k(all_scores, all_labels, k)

        return metrics

class RankingEvaluator:
    def __init__(self, model, ks=[5, 10, 20], device='cpu'):
        self.model = model
        self.device = device
        self.metrics = RankingMetrics(ks=ks)

    def evaluate(self, val_loader):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                user_embs = batch['user_embs'].to(self.device)
                item_embs = batch['item_embs'].to(self.device)
                labels = batch['labels']

                batch_size, num_items, embed_dim = item_embs.shape

                #flattening for model forward pass
                user_flat = user_embs.view(-1, embed_dim)
                item_flat = item_embs.view(-1, embed_dim)

                logits = self.model(user_flat, item_flat)

                #reshaping
                logits = logits.view(batch_size, num_items)

                self.metrics.update(logits, labels)

        return self.metrics.compute()
