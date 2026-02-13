import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class UserTower(nn.Module):
    def __init__(self, item_encoder, lstm_hidden_size):
        super().__init__()

        #the item tower to encode our user history
        self.item_encoder = item_encoder

        hidden_dim = item_encoder.fusion_mlp[-1].out_features

        #an lstm to handle the sequential nature of our data
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        #needed to project our lstm's output to the same latent space as our items
        self.projection = nn.Linear(in_features=lstm_hidden_size, out_features=hidden_dim)

    def forward(self, hist_text, hist_mask, hist_ids):
        batch_size, seq_len, text_dim = hist_text.shape

        hist_text = hist_text.reshape(-1, text_dim)

        hist_ids = hist_ids.reshape(-1)

        #passing as input to our item encoder
        flat_embeds = self.item_encoder(hist_text, hist_ids)

        embeds = flat_embeds.view(batch_size, seq_len, -1)

        embeds = embeds * hist_mask.unsqueeze(-1)

        lengths = hist_mask.sum(dim=1).cpu().long()

        lengths = torch.clamp(lengths, min=1)

        #to avoid computation on padding
        packed_embeds = pack_padded_sequence(input=embeds, lengths=lengths, batch_first=True, enforce_sorted=False)

        _, (h_n, c_n) = self.lstm(packed_embeds)

        final_state = h_n[-1]

        user_vector = self.projection(final_state)

        #for the same reasons as explained in the item tower
        return F.normalize(user_vector, p=2, dim=1)
        