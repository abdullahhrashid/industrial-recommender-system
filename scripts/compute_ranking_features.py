import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import pickle
import os
from tqdm import tqdm
from src.models.two_tower import TwoTowerModel
from src.data.retrieval_dataset import embeddings, idx_map
from src.utils.logging import get_logger
from torch.nn.utils.rnn import pad_sequence

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_metadata(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

#so that we don't have to compute this during training, helps with training time
def compute_user_embeddings(model, df, device, batch_size=2048):
    model.eval()
    
    histories = df['history_seq'].to_list()
    num_samples = len(histories)
    
    all_user_embeds = []
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_samples, batch_size), desc='Computing user embeddings'):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_histories = histories[start_idx:end_idx]
            
            batch_text_list = []
            batch_id_list = []
            
            for hist in batch_histories:
                hist_arr = np.array(hist)
                batch_text_list.append(torch.from_numpy(embeddings[hist_arr]).float())
                batch_id_list.append(torch.from_numpy(idx_map[hist_arr]).long())
            
            padded_text = pad_sequence(batch_text_list, batch_first=True, padding_value=0).to(device)
            padded_ids = pad_sequence(batch_id_list, batch_first=True, padding_value=0).to(device)
            
            mask = (padded_ids != 0).float()
            
            user_embeds = model.user_tower(padded_text, mask, padded_ids)
            
            all_user_embeds.append(user_embeds.cpu().numpy())
            
    return np.concatenate(all_user_embeds, axis=0)

def main(args):
    config = load_config(args.config)
    metadata = load_metadata(config['data']['metadata_path'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Running on {device}')
    
    model = TwoTowerModel(
        id_vocab_size=int(metadata['num_item_classes']),
        text_embed_dim=metadata['embed_dim'],
        id_embed_dim=config['model']['id_embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout_p=config['model']['dropout_p'],
        lstm_hidden_size=config['model']['lstm_hidden_size']
    ).to(device)
    
    logger.info(f'Loading checkpoint from {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    logger.info('Loading training data')
    train_df = pd.read_parquet(config['data']['train_path'])
    
    logger.info('Loading validation data')
    val_df = pd.read_parquet(config['data']['val_path'])

    test_path = config['data']['val_path'].replace('val', 'test')
    logger.info('Loading test data')
    test_df = pd.read_parquet(test_path)

    logger.info(f'Computing user embeddings for {len(train_df)} training samples...')
    train_user_embeds = compute_user_embeddings(model, train_df, device=device)
    
    logger.info(f'Computing user embeddings for {len(val_df)} validation samples...')
    val_user_embeds = compute_user_embeddings(model, val_df, device=device)

    logger.info(f'Computing user embeddings for {len(test_df)} validation samples...')
    test_user_embeds = compute_user_embeddings(model, test_df, device=device)
    
    output_dir = '../data/artifacts'
    os.makedirs(output_dir, exist_ok=True)
    
    train_save_path = os.path.join(output_dir, 'train_user_embeds.npy')
    val_save_path = os.path.join(output_dir, 'val_user_embeds.npy')
    test_save_path = os.path.join(output_dir, 'test_user_embeds.npy')
    
    np.save(train_save_path, train_user_embeds)
    np.save(val_save_path, val_user_embeds)
    np.save(test_save_path, test_user_embeds)
    
    logger.info(f'Saved training embeddings to {train_save_path} ({train_user_embeds.shape})')
    logger.info(f'Saved validation embeddings to {val_save_path} ({val_user_embeds.shape})')
    logger.info(f'Saved test embeddings to {test_save_path} ({test_user_embeds.shape})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute user embeddings for ranking')
    parser.add_argument('--config', type=str, required=True, help='Path to retrieval config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to retrieval model checkpoint')
    
    args = parser.parse_args()
    main(args)
