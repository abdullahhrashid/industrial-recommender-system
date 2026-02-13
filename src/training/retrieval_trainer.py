import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from src.data.retrieval_dataset import get_global_negatives, idx_map

#a class to help with training
class RetrievalTrainer:
    def __init__(self, model, optimizer, loss_fn, num_global_negs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_global_negs = num_global_negs

    #this function does all the magic
    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        total_loss = 0
        
        #a nice progress bar so that we don't stare at nothing during training
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch_idx}')

        for batch in pbar:
            self.optimizer.zero_grad()

            target_text = batch['target_text'].to(self.device)
            target_id = batch['target_id'].to(self.device)
            target_probs = batch['target_prob'].to(self.device)
            
            hist_text = batch['hist_text'].to(self.device)
            hist_id = batch['hist_id'].to(self.device)
            hist_mask = batch['hist_mask'].to(self.device)

            #running a pass through our model
            pos_item_embeds = self.model.item_tower(target_text, target_id)
            
            user_embeds = self.model.user_tower(hist_text, hist_mask, hist_id)
            
            #for mixed negative sampling
            random_indices = np.random.randint(0, len(idx_map), size=self.num_global_negs)

            global_neg_text, global_neg_ids = get_global_negatives(random_indices)
            
            global_neg_embeds = self.model.item_tower(global_neg_text.to(self.device), global_neg_ids.to(self.device))

            #calculating the loss
            loss = self.loss_fn(user_embeds=user_embeds, pos_item_embeds=pos_item_embeds, global_neg_embeds=global_neg_embeds, item_probs=target_probs)

            #backprop
            loss.backward()
            
            #clipping gradients to prevent exploding gradients in the lstm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            #updating weights
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(train_loader)
