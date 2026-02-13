import torch
import torch.nn as nn

class RankingModel(nn.Module):
    def __init__(self, embed_dim, hidden_dims, dropout_p):
        super().__init__()

        input_dim = embed_dim * 3

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim

        #final prediction layer is going to be a logit
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_emb, item_emb):
        #interaction feature
        hadamard = user_emb * item_emb

        #concatenate all features
        x = torch.cat([user_emb, item_emb, hadamard], dim=-1)

        return self.mlp(x).squeeze(-1)
