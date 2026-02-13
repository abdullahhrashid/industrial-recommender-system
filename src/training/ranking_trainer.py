import torch.nn as nn
from tqdm import tqdm

class RankingTrainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        total_loss = 0

        #a nice bar to visualize our progress
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch_idx}')

        for batch in pbar:
            self.optimizer.zero_grad()

            user_embs = batch['user_embs'].to(self.device)
            item_embs = batch['item_embs'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(user_embs, item_embs)

            loss = self.loss_fn(logits, labels)

            loss.backward()

            #so that grads don't explode
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        pbar.close()
        return total_loss / len(train_loader)
