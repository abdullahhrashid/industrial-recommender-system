import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.logging import get_logger

logger = get_logger(__name__)

class RankingDataset(Dataset):
    def __init__(self, user_embeds, interactions, item_embeds, candidates, max_candidates=None):
        super().__init__()

        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        #slice to top-N candidates for training (they're already sorted by FAISS score)
        if max_candidates and max_candidates < candidates.shape[1]:
            self.candidates = candidates[:, :max_candidates]
        else:
            self.candidates = candidates
        self.target_global_indices = interactions['target_global_idx'].values
        self.num_candidates = self.candidates.shape[1]

    def __len__(self):
        return len(self.target_global_indices)

    def __getitem__(self, idx):
        user_emb = self.user_embeds[idx]
        pos_item_idx = self.target_global_indices[idx]
        candidate_indices = self.candidates[idx].copy()

        #check if positive item is in the candidate set
        pos_mask = (candidate_indices == pos_item_idx)

        if pos_mask.any():
            pos_position = np.where(pos_mask)[0][0]
        else:
            #force-inject positive by replacing the last candidate
            candidate_indices[-1] = pos_item_idx
            pos_position = len(candidate_indices) - 1

        #get item embeddings for all candidates
        candidate_embeds = self.item_embeds[candidate_indices]

        #labels: 1 for positive, 0 for all others
        labels = np.zeros(self.num_candidates, dtype=np.float32)
        labels[pos_position] = 1.0

        return {
            'user_emb': torch.from_numpy(user_emb.copy()).float(),
            'item_embs': torch.from_numpy(candidate_embeds).float(),
            'labels': torch.from_numpy(labels).float()
        }


class RankingEvalDataset(Dataset):
    def __init__(self, user_embeds, interactions, item_embeds, candidates):
        super().__init__()

        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        self.candidates = candidates
        self.target_global_indices = interactions['target_global_idx'].values
        self.num_candidates = candidates.shape[1]

    def __len__(self):
        return len(self.target_global_indices)

    def __getitem__(self, idx):
        user_emb = self.user_embeds[idx]
        pos_item_idx = self.target_global_indices[idx]
        candidate_indices = self.candidates[idx].copy()

        pos_mask = (candidate_indices == pos_item_idx)

        if pos_mask.any():
            pos_position = np.where(pos_mask)[0][0]
        else:
            candidate_indices[-1] = pos_item_idx
            pos_position = len(candidate_indices) - 1

        candidate_embeds = self.item_embeds[candidate_indices]

        labels = np.zeros(self.num_candidates, dtype=np.float32)
        labels[pos_position] = 1.0

        return {
            'user_emb': torch.from_numpy(user_emb.copy()).float(),
            'item_embs': torch.from_numpy(candidate_embeds).float(),
            'labels': torch.from_numpy(labels).float()
        }


def ranking_collate_fn(batch):
    #single user embedding per sample (not tiled)
    user_embs = torch.stack([item['user_emb'] for item in batch])
    item_embs = torch.stack([item['item_embs'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    batch_size, num_candidates, embed_dim = item_embs.shape

    #expand user embeddings to match items (zero-copy)
    user_embs = user_embs.unsqueeze(1).expand(-1, num_candidates, -1)

    #flatten for model
    user_embs = user_embs.reshape(-1, embed_dim)
    item_embs = item_embs.reshape(-1, embed_dim)
    labels = labels.reshape(-1)

    return {
        'user_embs': user_embs,
        'item_embs': item_embs,
        'labels': labels
    }


def ranking_eval_collate_fn(batch):
    #keep grouped by user for ranking metrics
    user_embs = torch.stack([item['user_emb'] for item in batch])
    item_embs = torch.stack([item['item_embs'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'user_embs': user_embs,
        'item_embs': item_embs,
        'labels': labels
    }