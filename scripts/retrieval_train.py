from src.models.two_tower import TwoTowerModel
from src.models.loss import InfoNCE
from src.data.retrieval_dataset import RetrievalDataset, RetrievalEvalDataset, retrieval_collate_fn, eval_collate_fn, embeddings, idx_map
from src.training.retrieval_trainer import RetrievalTrainer
from src.evaluation.retrieval_evaluator import FullCatalogEvaluator
from src.utils.logging import get_logger
import argparse
import yaml
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import wandb
import os
import pickle
from dotenv import load_dotenv

#loading env variables
load_dotenv()

#intializing logger
logger = get_logger(__file__) 

#for loading config file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

#for loading metadata
def load_metadata(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

#main function
def main(args):
    config = load_config(args.config)
    metadata = load_metadata(config['data']['metadata_path'])

    logger.info('Loaded Metadata and Configs')
    
    id_vocab_size = int(metadata['num_item_classes'])
    text_embed_dim = metadata['embed_dim']
    
    #wandb initialization
    wandb.init(
        project=config['experiment']['project_name'],
        name=args.run_name if args.run_name else config['experiment']['run_name'],
        config=config
    )

    logger.info('Initialized Weights and Biases')
    
    #i unfortunatley do not possess a gpu, if you happen to run this code and have one, this is for you
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'Running on {device}')

    #loading the data
    train_df = pd.read_parquet(config['data']['train_path'])
    val_df = pd.read_parquet(config['data']['val_path'])

    logger.info(f'Loaded Data')
    
    #creating pytorch datasets
    train_dataset = RetrievalDataset(train_df)
    val_dataset = RetrievalEvalDataset(val_df, embeddings, idx_map)

    #creating pytorch dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=retrieval_collate_fn,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['eval_batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=config['data']['num_workers']
    )

    logger.info('Created Dataloaders')

    #instantiating the model
    model = TwoTowerModel(
        id_vocab_size=id_vocab_size,
        text_embed_dim=text_embed_dim,
        id_embed_dim=config['model']['id_embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout_p=config['model']['dropout_p'],
        lstm_hidden_size=config['model']['lstm_hidden_size']
    ).to(device)

    model = torch.compile(model, mode='reduce-overhead')
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    loss_fn = InfoNCE(temperature=config['training']['temperature'])

    #setting up trainer and evaluator
    trainer = RetrievalTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, num_global_negs=config['training']['num_global_negs'])
    
    evaluator = FullCatalogEvaluator(model=model, embeddings=embeddings, idx_map=idx_map)

    #folder for saving model checkpoints
    save_dir = os.path.join('..', 'checkpoints', args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f'Saving checkpoints to: {os.path.abspath(save_dir)}')
    
    #training loop starts here
    best_recall = 0.0
    best_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = config['training'].get('early_stop_patience', 5)

    for epoch in range(config['training']['num_epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        
        #step lr scheduler based on training loss (every epoch)
        scheduler.step(train_loss)
        
        # Current learning rate (after scheduler step)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log training metrics
        logger.info(f'Epoch {epoch+1} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}')
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr
        })
        
        #early stopping based on training loss
        if train_loss < best_loss:
            best_loss = train_loss
            epochs_without_improvement = 0
            logger.info(f'New best training loss: {best_loss:.4f}')
        else:
            epochs_without_improvement += 1
            logger.info(f'No improvement in training loss for {epochs_without_improvement} epoch(s)')
            
            if epochs_without_improvement >= early_stop_patience:
                logger.info(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        #evaluating only at the end of training (if you have a good gpu, you can do this at every epoch)
        if (epoch + 1) == config['training']['num_epochs'] or epochs_without_improvement >= early_stop_patience:
            logger.info('Running final evaluation...')
            
            metrics = evaluator.evaluate(val_loader)
            val_recall = metrics['Recall@20']

            del evaluator.all_item_embeds
            evaluator.all_item_embeds = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            #formatting metrics for logging
            metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            logger.info(f'Final Evaluation Metrics: {metrics_str}')

            wandb.log({
                'final_epoch': epoch + 1,
                **{f'val_{k.lower().replace("@", "_at_")}': v for k, v in metrics.items()}
            })

            #saving final model
            save_path = os.path.join(save_dir, 'best_retrieval_model.pth')
            torch.save(model.state_dict(), save_path)
            logger.info('Final Model Saved')

    #saving model to wandb
    wandb.save(save_path, base_path=save_dir)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Two-Tower Retrieval Model')
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--run_name', type=str, required=True, help='Name for WandB run')
    
    args = parser.parse_args()
    main(args)
    