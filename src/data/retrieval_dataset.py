import pandas as pd
from src.utils.logging import get_logger
from torch.utils.data import Dataset
import os
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

#setting up the logger
logger = get_logger(__file__)

#paths to artifacts
embedding_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/embedding.npy')
global_idx_to_model_idx_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/global_to_warm_map.npy')

embeddings = np.load(embedding_path)
idx_map = np.load(global_idx_to_model_idx_path)

class RetrievalDataset(Dataset):
    def __init__(self, interactions):
        super().__init__()

        #native arrays/lists are faster than pandas columns
        self.user_histories = interactions['history_seq'].to_list()
        self.target = interactions['target_global_idx'].values

        #keeps the code from crashing if i send in the test or validation set
        if 'target_prob' in interactions.columns:
            self.target_probs = interactions['target_prob'].values
        else:
            #dummy value
            self.target_probs = np.ones(len(interactions), dtype=np.float32)

    def __len__(self):
        return len(self.user_histories)

    def __getitem__(self, idx):
        user_history = np.array(self.user_histories[idx])
        target = self.target[idx]
        
        embedding_history = embeddings[user_history]
        target_embedding = embeddings[target]

        id_history = idx_map[user_history]
        target_id = idx_map[target]

        target_prob = self.target_probs[idx]

        return {
            #inputs
            'input_embeddings': torch.from_numpy(embedding_history.copy()).float(),
            'input_ids': torch.from_numpy(id_history).long(),
            #targets
            'target_embedding':torch.from_numpy(target_embedding.copy()).float(),
            'target_id': torch.tensor(target_id, dtype=torch.long),
            #probabilties for log q correction
            'target_prob': torch.tensor(target_prob, dtype=torch.float)
        }
    
#helper for global negative sampling
def get_global_negatives(global_indices):
    #because i need to be on the cpu for numpy indexing
    if isinstance(global_indices, torch.Tensor):
        global_indices = global_indices.cpu().numpy()
        
    embeds = embeddings[global_indices]
    ids = idx_map[global_indices]
        
    return (torch.from_numpy(embeds).float(), torch.from_numpy(ids).long())

logger.info('Created PyTorch Dataset for Retrieval Phase and added Global Negative Sampling Capability')

#a collate function to handle padding and masking 
def retrieval_collate_fn(batch):
    hist_texts = [item['input_embeddings'] for item in batch]
    hist_ids = [item['input_ids'] for item in batch]
    
    target_texts = torch.stack([item['target_embedding'] for item in batch])
    target_ids = torch.stack([item['target_id'] for item in batch])
    
    target_probs = torch.stack([item['target_prob'] for item in batch])

    #padding
    padded_texts = pad_sequence(hist_texts, batch_first=True, padding_value=0)
    padded_ids = pad_sequence(hist_ids, batch_first=True, padding_value=0)

    #so that padding doesnt contribute to the loss
    mask = (padded_ids != 0).float()

    return {
        'hist_text': padded_texts,   
        'hist_id': padded_ids,       
        'hist_mask': mask,           
        'target_text': target_texts, 
        'target_id': target_ids,
        'target_prob': target_probs 
    }

class RetrievalEvalDataset(Dataset):
    def __init__(self, interactions, embeddings, idx_map):
        self.histories = interactions['history_seq'].to_list()
        self.targets = interactions['target_global_idx'].values
        self.embeddings = embeddings
        self.idx_map = idx_map

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        hist = np.array(self.histories[idx])
        target = self.targets[idx]
        
        return {
            'hist_text': torch.from_numpy(self.embeddings[hist].copy()).float(),
            'hist_id': torch.from_numpy(self.idx_map[hist].copy()).long(),
            'target_text': torch.from_numpy(self.embeddings[target].copy()).float(),
            'target_id': torch.tensor(self.idx_map[target], dtype=torch.long), 
            'target_global_idx': target,
            'seen_items': hist
        }
    
def eval_collate_fn(batch):
    hist_texts = [item['hist_text'] for item in batch]
    hist_ids = [item['hist_id'] for item in batch]
    
    padded_texts = pad_sequence(hist_texts, batch_first=True, padding_value=0)
    padded_ids = pad_sequence(hist_ids, batch_first=True, padding_value=0)
    
    mask = (padded_ids != 0).float()

    target_text = torch.stack([item['target_text'] for item in batch])
    target_id = torch.stack([item['target_id'] for item in batch])
    
    target_global_indices = torch.tensor([item['target_global_idx'] for item in batch], dtype=torch.long)
    
    seen_items = [item['seen_items'] for item in batch]

    return {
        'hist_text': padded_texts,
        'hist_id': padded_ids,
        'hist_mask': mask,
        'target_text': target_text,
        'target_id': target_id,
        'target_global_idx': target_global_indices,
        'seen_items': seen_items
    }
