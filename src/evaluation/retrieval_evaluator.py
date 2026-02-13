import torch
from tqdm import tqdm
from src.evaluation.retrieval_metrics import RetrievalMetrics
import numpy as np

#to be used if evaluation done at every epoch
class SubsetRetrievalEvaluator:
    def __init__(self, model, all_global_ids, embeddings, idx_map, num_negatives=5000):
        self.model = model
        self.all_global_ids = all_global_ids 
        self.embeddings = embeddings
        self.idx_map = idx_map 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_negatives = num_negatives
        
        self.negative_embeds = None 
        self.metrics = RetrievalMetrics()

    def update_negative_index(self):
        self.model.eval()
        
        perm = torch.randperm(len(self.all_global_ids))
        sampled_global_ids = self.all_global_ids[perm[:self.num_negatives]]

        sampled_global_ids_np = sampled_global_ids.cpu().numpy()
        neg_text_feats = torch.from_numpy(self.embeddings[sampled_global_ids_np]).float().to(self.device)
        
        neg_model_ids_np = self.idx_map[sampled_global_ids_np]
        neg_ids = torch.from_numpy(neg_model_ids_np).long().to(self.device)

        with torch.no_grad():
            self.negative_embeds = self.model.item_tower(neg_text_feats, neg_ids)

    def evaluate(self, val_loader):
        self.model.eval()
        self.metrics.reset()
        
        self.update_negative_index()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                hist_text = batch['hist_text'].to(self.device)
                hist_id = batch['hist_id'].to(self.device)
                hist_mask = batch['hist_mask'].to(self.device)
                
                user_embeds = self.model.user_tower(hist_text, hist_mask, hist_id)

                target_text = batch['target_text'].to(self.device) 
                target_id = batch['target_id'].to(self.device)
                
                pos_embeds = self.model.item_tower(target_text, target_id)

                pos_scores = (user_embeds * pos_embeds).sum(dim=1, keepdim=True)
                
                neg_scores = torch.matmul(user_embeds, self.negative_embeds.T)
                
                scores = torch.cat([pos_scores, neg_scores], dim=1)
                
                target_indices = torch.zeros(scores.size(0), dtype=torch.long, device=self.device)
                
                self.metrics.update(scores, target_indices)
        
        return self.metrics.compute()

#to be used if evaluation done at the end of a run
class FullCatalogEvaluator:    
    def __init__(self, model, embeddings, idx_map, batch_size=4096):
        self.model = model
        self.embeddings = embeddings
        self.idx_map = idx_map
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.metrics = RetrievalMetrics()
        
        self.all_item_embeds = None

    def build_item_index(self):
        self.model.eval()
        
        num_items = len(self.embeddings)
        all_embeds = []
        
        with torch.no_grad():
            for start in tqdm(range(0, num_items, self.batch_size), desc='Building item index'):
                end = min(start + self.batch_size, num_items)
                
                text_feats = torch.from_numpy(self.embeddings[start:end].copy()).float().to(self.device)
                ids = torch.from_numpy(self.idx_map[start:end].copy()).long().to(self.device)
                
                embeds = self.model.item_tower(text_feats, ids)
                all_embeds.append(embeds)
        
        self.all_item_embeds = torch.cat(all_embeds, dim=0)

    def evaluate(self, val_loader):
        self.model.eval()
        self.metrics.reset()
        
        self.build_item_index()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                hist_text = batch['hist_text'].to(self.device)
                hist_id = batch['hist_id'].to(self.device)
                hist_mask = batch['hist_mask'].to(self.device)
                target_global_idx = batch['target_global_idx']
                
                user_embeds = self.model.user_tower(hist_text, hist_mask, hist_id)
                
                scores = torch.matmul(user_embeds, self.all_item_embeds.T)
                
                self.metrics.update(scores, target_global_idx.to(self.device))
        
        return self.metrics.compute()

#comprehensive evaluator with accuracy and beyond-accuracy metrics
class TestEvaluator:
    def __init__(self, model, embeddings, idx_map, k_values=[10, 20, 50, 100, 500, 1000, 1500, 2000], batch_size=4096):
        self.model = model
        self.embeddings = embeddings
        self.idx_map = idx_map
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.k_values = k_values
        self.batch_size = batch_size
        self.all_item_embeds = None
        
    def build_item_index(self):
        #pre computing embeddings for all items
        self.model.eval()
        num_items = len(self.embeddings)
        all_embeds = []
        
        with torch.no_grad():
            pbar = tqdm(range(0, num_items, self.batch_size), desc='Building item index')
            for start in pbar:
                end = min(start + self.batch_size, num_items)
                text_feats = torch.from_numpy(self.embeddings[start:end].copy()).float().to(self.device)
                ids = torch.from_numpy(self.idx_map[start:end].copy()).long().to(self.device)
                embeds = self.model.item_tower(text_feats, ids)
                all_embeds.append(embeds.cpu())
            pbar.close()
        
        self.all_item_embeds = torch.cat(all_embeds, dim=0).to(self.device)

    #running full evaluation on test set
    def evaluate(self, test_loader, item_popularity=None):
        self.model.eval()
        self.build_item_index()
        
        #storage for metrics
        recalls = {k: [] for k in self.k_values}
        ndcgs = {k: [] for k in self.k_values}
        mrrs = []
        
        #beyond accuracy metrics
        all_recommended_items = set()
        intra_list_distances = {k: [] for k in self.k_values}
        novelty_scores = {k: [] for k in self.k_values}
        
        total_items = len(self.embeddings)
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating on test set')
            for batch in pbar:
                hist_text = batch['hist_text'].to(self.device)
                hist_id = batch['hist_id'].to(self.device)
                hist_mask = batch['hist_mask'].to(self.device)
                target_global_idx = batch['target_global_idx']
                
                user_embeds = self.model.user_tower(hist_text, hist_mask, hist_id)
                
                #scores against all items
                scores = torch.matmul(user_embeds, self.all_item_embeds.T)
                
                #get rankings
                _, top_indices = torch.topk(scores, k=max(self.k_values), dim=1)
                top_indices = top_indices.cpu().numpy()
                target_global_idx = target_global_idx.numpy()
                
                #calculate metrics for each user in batch
                for i in range(len(target_global_idx)):
                    target = target_global_idx[i]
                    
                    #find rank of target item
                    target_score = scores[i, target].item()
                    rank = (scores[i] > target_score).sum().item() + 1
                    
                    #mrr
                    mrrs.append(1.0 / rank)
                    
                    for k in self.k_values:
                        top_k = top_indices[i, :k]
                        
                        #recall
                        if target in top_k:
                            recalls[k].append(1.0)
                            #ndcg  find position
                            pos = np.where(top_k == target)[0][0]
                            ndcgs[k].append(1.0 / np.log2(pos + 2))
                        else:
                            recalls[k].append(0.0)
                            ndcgs[k].append(0.0)
                        
                        #coverage track all recommended items
                        all_recommended_items.update(top_k.tolist())
                        
                        #intralist diversity (average pairwise distance)
                        if k <= 100:
                            top_k_embeds = self.all_item_embeds[top_k].cpu().numpy()
                            if len(top_k_embeds) > 1:
                                #cosine similarity matrix
                                sim_matrix = np.dot(top_k_embeds, top_k_embeds.T)
                                #average off diagonal similarity
                                mask = ~np.eye(len(top_k_embeds), dtype=bool)
                                avg_sim = sim_matrix[mask].mean()
                                #diversity = 1 - similarity
                                intra_list_distances[k].append(1 - avg_sim)
                        
                        #novelty (average negative log popularity)
                        if item_popularity is not None:
                            pops = [item_popularity.get(idx, 1e-10) for idx in top_k]
                            novelty = np.mean([-np.log2(p + 1e-10) for p in pops])
                            novelty_scores[k].append(novelty)
            
            pbar.close()
        
        #compiling results
        results = {}
        
        #accuracy metrics
        results['MRR'] = np.mean(mrrs)
        for k in self.k_values:
            results[f'Recall@{k}'] = np.mean(recalls[k])
            results[f'NDCG@{k}'] = np.mean(ndcgs[k])
        
        #coverage
        results['Catalog_Coverage'] = len(all_recommended_items) / total_items
        results['Unique_Items_Recommended'] = len(all_recommended_items)
        results['Total_Items_in_Catalog'] = total_items
        
        #diversity
        for k in self.k_values:
            if k <= 50 and intra_list_distances[k]:
                results[f'Diversity@{k}'] = np.mean(intra_list_distances[k])
        
        #novelty
        for k in self.k_values:
            if novelty_scores[k]:
                results[f'Novelty@{k}'] = np.mean(novelty_scores[k])
        
        return results