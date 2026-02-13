import pandas as pd 
import os
import numpy as np
from src.utils.logging import get_logger
import pickle

logger = get_logger(__file__)

#paths to data
item_path = os.path.join(os.path.dirname(__file__), '../../data/interim/items.csv')
train_path = os.path.join(os.path.dirname(__file__), '../../data/interim/train.parquet')
val_path = os.path.join(os.path.dirname(__file__), '../../data/interim/val.parquet')
test_path = os.path.join(os.path.dirname(__file__), '../../data/interim/test.parquet')
mapping_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/text_embedding_meta.pkl')
embedding_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/text_embeddings.npy')

#paths to store data
final_embedding_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/embedding.npy')
bridge_map_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/global_to_warm_map.npy')
final_meta_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/final_metadata.pkl')

#reading the data
items = pd.read_csv(item_path)
train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)
test = pd.read_parquet(test_path)

with open(mapping_path, 'rb') as file:
    obj = pickle.load(file)

#i realized i need 0 for padding, so 1 is reserved for warm items now
items['item_idx'] = items['item_idx'] + 1 
train['target_item_idx'] = train['target_item_idx'] + 1
val['target_item_idx'] = val['target_item_idx'] + 1
test['target_item_idx'] = test['target_item_idx'] + 1

raw_mapping = obj['asin_to_row']

mapping = {asin:idx+1 for asin, idx in raw_mapping.items()}

reverse_mapping = {idx: asin for asin, idx in mapping.items()}

#converting asins to a global index, helps in training efficiency
items['global_idx'] = items['parent_asin'].map(mapping).astype(int)
train['target_global_idx'] = train['target_parent_asin'].map(mapping).astype(int)
val['target_global_idx'] = val['target_parent_asin'].map(mapping).astype(int)
test['target_global_idx'] = test['target_parent_asin'].map(mapping).astype(int)

#a function for converting a history of asins to a history of global indices
lookup_func = lambda history: [mapping[asin] for asin in history if asin in mapping]

#applying to the columns
train['history_seq'] = train['history_asins'].apply(lookup_func)
val['history_seq'] = val['history_asins'].apply(lookup_func)
test['history_seq'] = test['history_asins'].apply(lookup_func)

logger.info('Transformed columns to utilize Global Item Indices')

#loading the embeddings
raw_embeddings = np.load(embedding_path)

embedding_dim = obj['embedding_dim']

#we will use a null vector for the padding
zero_vector = np.zeros((1, embedding_dim), dtype=raw_embeddings.dtype)

#adding padding to the embeddings
final_embeddings = np.vstack([zero_vector, raw_embeddings])

#saving the final embeddings
np.save(final_embedding_path, final_embeddings)

logger.info('Updated embeddings to accommodate padding')

#i will use this to load the ids utilized by the models using the global index
max_global_idx = items['global_idx'].max()

bridge_map = np.zeros(max_global_idx + 1, dtype=np.int32)

items_sorted = items.sort_values('global_idx')

bridge_map[items_sorted['global_idx'].values] = items_sorted['item_idx'].values

#saving the map
np.save(bridge_map_path, bridge_map)

#saving stuff that will help us at inference time
final_metadata = {
    'asin_to_global_idx': mapping,
    'global_idx_to_asin': reverse_mapping,
    'vocab_size': len(mapping) + 1, 
    'embed_dim': embedding_dim,
    'embedding_model_name': 'all-mpnet-base-v2',
    'num_item_classes': items['item_idx'].max() + 1
}

#storing it as a pickle file
with open(final_meta_path, 'wb') as f:
    pickle.dump(final_metadata, f)

#dropping these unneeded columns
cols_to_drop = ['history_asins', 'target_parent_asin']

train = train.drop(columns=cols_to_drop)
val = val.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

#paths to save data
item_path = os.path.join(os.path.dirname(__file__), '../../data/processed/items.csv')
train_path = os.path.join(os.path.dirname(__file__), '../../data/processed/train.parquet')
val_path = os.path.join(os.path.dirname(__file__), '../../data/processed/val.parquet')
test_path = os.path.join(os.path.dirname(__file__), '../../data/processed/test.parquet')

#saving
train.to_parquet(train_path, index=False)
val.to_parquet(val_path, index=False)
test.to_parquet(test_path, index=False)
items.to_csv(item_path, index=False)

logger.info('Saved metadata and final datasets')
