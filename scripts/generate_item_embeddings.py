from src.models.two_tower import TwoTowerModel
from src.data.retrieval_dataset import embeddings, idx_map
from src.utils.logging import get_logger
import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import pickle

logger = get_logger(__file__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_metadata(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def main(args):
    config = load_config(args.config)
    metadata = load_metadata(config['data']['metadata_path'])

    logger.info('Loaded config and metadata')
    
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

    checkpoint_path = args.checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        logger.info('Detected torch.compile() checkpoint, stripped _orig_mod. prefix')
    
    model.load_state_dict(state_dict)
    
    logger.info(f'Loaded model from {checkpoint_path}')

    item_tower = model.item_tower
    item_tower.eval()  

    logger.info('Starting item embedding export...')
    
    items_df = pd.read_csv('../data/processed/items.csv', usecols=['title', 'parent_asin', 'global_idx'])
    logger.info(f'Loaded items metadata with {len(items_df)} items')
    
    num_items = len(embeddings)
    logger.info(f'Processing {num_items} items')
    
    batch_size = 2048
    all_item_embeds = []
    
    with torch.no_grad():
        for start_idx in range(0, num_items, batch_size):
            end_idx = min(start_idx + batch_size, num_items)
            batch_indices = np.arange(start_idx, end_idx)
            
            batch_text = torch.from_numpy(embeddings[batch_indices]).float().to(device)
            batch_ids = torch.from_numpy(idx_map[batch_indices]).long().to(device)
            
            batch_embeds = item_tower(batch_text, batch_ids)
            all_item_embeds.append(batch_embeds.cpu().numpy())
            
            if (start_idx // batch_size) % 10 == 0:
                logger.info(f'Processed {end_idx}/{num_items} items')
    
    all_item_embeds = np.concatenate(all_item_embeds, axis=0)
    logger.info(f'Generated embeddings shape: {all_item_embeds.shape}')
    
    metadata_map = {}
    for _, row in items_df.iterrows():
        global_idx = int(row['global_idx'])
        if global_idx < num_items:
            metadata_map[global_idx] = {
                'item_id': row['parent_asin'],
                'title': row['title'],
                'global_idx': global_idx
            }
    
    output_dir = '../data/artifacts'
    embeddings_path = f'{output_dir}/faiss_item_embeddings.npy'
    metadata_path = f'{output_dir}/faiss_metadata.pkl'
    
    np.save(embeddings_path, all_item_embeds)
    logger.info(f'Saved item embeddings to {embeddings_path}')
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_map, f)
    logger.info(f'Saved metadata to {metadata_path}')
    
    logger.info('Item embedding export complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Item Embeddings for FAISS Retrieval')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    args = parser.parse_args()
    main(args)
