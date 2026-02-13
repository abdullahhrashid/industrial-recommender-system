from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
from src.utils.logging import get_logger
import joblib
import os

#setting up the logger
logger = get_logger(__file__)

#path to items
item_path = os.path.join(os.path.dirname(__file__), '../../data/interim/products.csv')

items = pd.read_csv(item_path)

#to avoid any issues becuase im going to use this order of indices as a map
items = items.reset_index(drop=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#a good pretrained encoder
model = SentenceTransformer('all-mpnet-base-v2', device=device)

#generating embeddings and storing them in a numpy array
embeddings = model.encode(items['input_text'].tolist(), batch_size=480, convert_to_numpy=True, normalize_embeddings=True)

logger.info('Created text embeddings for the input')

#index map
asin_to_row = {asin: idx for idx, asin in enumerate(items['parent_asin'])}

#paths for artifacts
embedding_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/text_embeddings.npy')
index_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/text_embedding_meta.pkl')

#saving embeddings
np.save(embedding_path, embeddings)

#saving index map
joblib.dump(
    {
        'asin_to_row': asin_to_row,
        'embedding_dim': embeddings.shape[1],
        'model_name': 'all-mpnet-base-v2'
    },
    index_path
)

logger.info('Saved metadata and embeddings')
