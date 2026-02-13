import torch
import numpy as np
import faiss
import pickle
import yaml
from src.models.two_tower import TwoTowerModel
from src.data.retrieval_dataset import embeddings, idx_map
from src.utils.logging import get_logger

logger = get_logger(__name__)

#pre computing candidates for a user to speed up training
class CandidateGenerator:
    def __init__(self, model_path, faiss_index_path, metadata_path, config_path, device):
        self.device = device
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
 
        logger.info(f'Loading metadata from {metadata_path}')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        meta_path = self.config['data']['metadata_path']
        with open(meta_path, 'rb') as f:
            model_metadata = pickle.load(f)
        
        logger.info(f'Initializing model')

        self.model = TwoTowerModel(
            id_vocab_size=int(model_metadata['num_item_classes']),
            text_embed_dim=model_metadata['embed_dim'],
            id_embed_dim=self.config['model']['id_embed_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout_p=self.config['model']['dropout_p'],
            lstm_hidden_size=self.config['model']['lstm_hidden_size']
        ).to(device)
        
        logger.info(f'Loading checkpoint from {model_path}')
        state_dict = torch.load(model_path, map_location=device)
        
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f'Loading FAISS index from {faiss_index_path}')
        self.index = faiss.read_index(faiss_index_path)
        
        logger.info(f'Candidate Generator initialized with {self.index.ntotal} items')
    
    def encode_user_history(self, history_indices):
        with torch.no_grad():
            hist_text = torch.from_numpy(embeddings[history_indices]).float().to(self.device)
            hist_ids = torch.from_numpy(idx_map[history_indices]).long().to(self.device)
            
            hist_text = hist_text.unsqueeze(0) 
            hist_ids = hist_ids.unsqueeze(0)    
            
            mask = torch.ones_like(hist_ids, dtype=torch.float32)
            
            user_emb = self.model.user_tower(hist_text, mask, hist_ids)
            
            return user_emb.cpu().numpy()  
    
    def generate_candidates(self, user_history, k, exclude_seen=True):
        history_arr = np.array(user_history, dtype=np.int64)
        
        user_emb = self.encode_user_history(history_arr)
        
        user_emb = user_emb.astype('float32')
        
        search_k = k * 3 if exclude_seen else k
        scores, indices = self.index.search(user_emb, search_k)
        
        scores = scores[0]
        indices = indices[0]
        
        candidates = []
        seen_set = set(user_history) if exclude_seen else set()
        
        for idx, score in zip(indices, scores):
            if idx == -1: 
                break
                
            if exclude_seen and idx in seen_set:
                continue
            
            meta = self.metadata.get(idx, {})
            
            candidates.append({
                'global_idx': int(idx),
                'item_id': meta.get('item_id', 'unknown'),
                'title': meta.get('title', 'Unknown'),
                'score': float(score),
                'rank': len(candidates) + 1
            })
            
            if len(candidates) >= k:
                break
        
        return candidates
    
    def batch_generate_candidates(self, user_histories, k, exclude_seen=True):
        results = []
        for history in user_histories:
            candidates = self.generate_candidates(history, k, exclude_seen)
            results.append(candidates)
        
        return results
