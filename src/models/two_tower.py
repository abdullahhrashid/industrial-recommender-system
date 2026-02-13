from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, id_vocab_size, text_embed_dim, id_embed_dim, hidden_dim, dropout_p, lstm_hidden_size):
        super().__init__()

        #initializing the item tower 
        self.item_tower = ItemTower(id_vocab_size, text_embed_dim, id_embed_dim, hidden_dim, dropout_p)

        #initializing the user tower
        self.user_tower = UserTower(self.item_tower, lstm_hidden_size)

    def forward(self, batch):
        target_embedding = self.item_tower(batch['target_text'], batch['target_id'])
        
        user_embedding = self.user_tower(batch['hist_text'], batch['hist_mask'], batch['hist_id'])
        
        #the two embeddings we are basing our model on!
        return user_embedding, target_embedding
