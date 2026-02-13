import torch
import torch.nn as nn
import torch.nn.functional as F

#the architecture that will produce embeddings for our items
class ItemTower(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, id_embed_dim, hidden_dim, id_dropout_p):
        super().__init__()

        self.id_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=id_embed_dim, padding_idx=0)
        
        self.text_projection = nn.Linear(in_features=text_embed_dim, out_features=id_embed_dim)

        #we will use this to fuse the two signals into one singal representation
        self.gate = nn.Sequential(nn.Linear(2 * id_embed_dim, 1),
                                  nn.Sigmoid())

        #this will also help with fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(id_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.id_dropout_p = id_dropout_p
        self.cold_item_idx = 1

        #so that we dont get any signal from our id if it is cold
        with torch.no_grad():
            self.id_embedding.weight[self.cold_item_idx].zero_()

        def freeze_cold_item_grad(grad):
            grad[self.cold_item_idx].zero_()
            return grad

        self.id_embedding.weight.register_hook(freeze_cold_item_grad)

    def forward(self, text_feat, item_ids):
        #so that our item tower doesn't only depend on the ids, this is for cold items, we will use the signal from their text
        if self.training and self.id_dropout_p > 0:
            mask = torch.rand_like(item_ids, dtype=torch.float) < self.id_dropout_p
            item_ids = item_ids.clone()
            item_ids[mask] = self.cold_item_idx

        id_embeds = self.id_embedding(item_ids)
        text_projs = self.text_projection(text_feat)
        
        #fusing the two into one representation
        fusion_input = torch.cat([id_embeds, text_projs], dim=-1)
        gate = self.gate(fusion_input)
        fusion_input = gate * id_embeds + (1-gate) * text_projs

        output = self.fusion_mlp(fusion_input)

        #because we don't want our model to learn to just output very large vectors to increase similarity score
        return F.normalize(output, p=2, dim=-1)
