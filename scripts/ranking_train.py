from src.models.ranking import RankingModel
from src.data.ranking_dataset import RankingDataset, RankingEvalDataset, ranking_collate_fn, ranking_eval_collate_fn
from src.training.ranking_trainer import RankingTrainer
from src.evaluation.ranking_evaluator import RankingEvaluator
from src.utils.logging import get_logger
import argparse
import yaml
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__file__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    config = load_config(args.config)

    logger.info('Loaded config')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Running on {device}')

    wandb.init(project=config['experiment']['project_name'], name=args.run_name if args.run_name else 'ranking-run', config=config)

    logger.info('Initialized Weights and Biases')

    #loading pre computed embeddings
    item_embeds = np.load(config['data']['item_embeds_path'])
    train_user_embeds = np.load(config['data']['train_user_embeds_path'])
    val_user_embeds = np.load(config['data']['val_user_embeds_path'])

    logger.info(f'Loaded item embeddings: {item_embeds.shape}')
    logger.info(f'Loaded train user embeddings: {train_user_embeds.shape}')
    logger.info(f'Loaded val user embeddings: {val_user_embeds.shape}')

    #loading pre computed faiss candidates
    train_candidates = np.load(config['data']['train_candidates_path'])
    val_candidates = np.load(config['data']['val_candidates_path'])

    logger.info(f'Loaded train candidates: {train_candidates.shape}')
    logger.info(f'Loaded val candidates: {val_candidates.shape}')

    #loading interaction data 
    train_df = pd.read_parquet(config['data']['train_path'])
    val_df = pd.read_parquet(config['data']['val_path'])

    logger.info(f'Loaded interaction data: train={len(train_df)}, val={len(val_df)}')

    #creating datasets
    train_dataset = RankingDataset(
        user_embeds=train_user_embeds,
        interactions=train_df,
        item_embeds=item_embeds,
        candidates=train_candidates,
        max_candidates=config['training'].get('max_candidates')
    )

    val_dataset = RankingEvalDataset(
        user_embeds=val_user_embeds,
        interactions=val_df,
        item_embeds=item_embeds,
        candidates=val_candidates
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=ranking_collate_fn,
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['eval_batch_size'],
        shuffle=False,
        collate_fn=ranking_eval_collate_fn,
        num_workers=config['data']['num_workers']
    )

    logger.info('Created dataloaders')

    #instantiating the model
    model = RankingModel(
        embed_dim=config['model']['embed_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_p=config['model']['dropout_p']
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    trainer = RankingTrainer(model=model, optimizer=optimizer, device=device)
    evaluator = RankingEvaluator(model=model, device=device)

    #checkpoint directory
    save_dir = os.path.join('..', 'checkpoints', args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f'Saving checkpoints to: {os.path.abspath(save_dir)}')

    #training loop
    #early stopping params
    best_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = max(config['training'].get('early_stop_patience', 3), scheduler.patience + 1)
    
    #training loop
    for epoch in range(config['training']['num_epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)

        #scheduler step based on training loss
        scheduler.step(train_loss)

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f'Epoch {epoch+1} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}')

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr
        })

        #freeing fragmented memory to prevent progressive slowdown
        gc.collect()
        
        #early stopping logic
        if train_loss < best_loss:
            best_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info(f'Early stopping triggered after epoch {epoch+1}')
                break

    #evaluatung
    logger.info('Training complete, running final evaluation...')

    metrics = evaluator.evaluate(val_loader)

    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logger.info(f'Final Metrics: {metrics_str}')

    wandb.log({
        **{f'val_{k.lower().replace("@", "_at_")}': v for k, v in metrics.items()}
    })

    #saving final model
    save_path = os.path.join(save_dir, 'best_ranking_model.pth')
    torch.save(model.state_dict(), save_path)
    logger.info(f'Saved ranking model to {save_path}')

    #logging stuff to wandb
    wandb.save(save_path, base_path=save_dir)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Ranking Model')
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--run_name', type=str, required=True, help='Name for WandB run')

    args = parser.parse_args()
    main(args)
