import torch
import torch.nn as nn

class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super().__init__()

        #hyperparameter for the loss function
        self.temperature = temperature

        #this loss function utilizes the cross entropy loss in a really cool way
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, user_embeds, pos_item_embeds, global_neg_embeds, item_probs=None): # item_probs not needed
        device = user_embeds.device
        batch_size = user_embeds.shape[0]

        pos_logits = (user_embeds * pos_item_embeds).sum(dim=1, keepdim=True)

        #in batch negatives
        in_batch_logits = torch.matmul(user_embeds, pos_item_embeds.T)
        
        #mask diagonal
        mask = torch.eye(batch_size, device=device).bool()
        in_batch_neg_logits = in_batch_logits.masked_fill(mask, -1e9)

        #global negatives
        global_logits = torch.matmul(user_embeds, global_neg_embeds.T)
            
        #concatenating
        neg_logits = torch.cat([in_batch_neg_logits, global_logits], dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        #temperature and loss
        logits = logits / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        return self.cross_entropy(logits, labels)    
    