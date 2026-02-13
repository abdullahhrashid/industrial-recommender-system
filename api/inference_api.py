import torch
import numpy as np
import faiss
import pickle
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException
import time
from contextlib import asynccontextmanager
from src.models.two_tower import TwoTowerModel
from src.models.ranking import RankingModel
from src.data.retrieval_dataset import embeddings, idx_map
from src.utils.logging import get_logger

logger = get_logger(__name__)

#global state for loaded models and data
state = {}

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_user_history_map(config):
    history_map = {}
    title_map = {}

    #load item titles
    items_df = pd.read_csv(config['data']['items_path'])
    for _, row in items_df.iterrows():
        title_map[int(row['global_idx'])] = {
            'title': row['title'],
            'item_id': row['parent_asin']
        }

    #aggregate history from all splits
    for split_key in ['train_path', 'val_path', 'test_path']:
        path = config['data'].get(split_key)
        if path:
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                uid = int(row['user_idx'])
                target = int(row['target_global_idx'])

                if uid not in history_map:
                    history_map[uid] = []

                #add history items
                for item in row['history_seq']:
                    if item not in history_map.get(uid, []):
                        history_map[uid].append(int(item))

                #add the target item too
                if target not in history_map[uid]:
                    history_map[uid].append(target)

    return history_map, title_map

def load_retrieval_model(config, device):
    retrieval_config = load_config(config['retrieval']['config_path'])

    meta_path = retrieval_config['data']['metadata_path']

    if meta_path.startswith('../'):
        meta_path = meta_path[3:]

    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    model = TwoTowerModel(
        id_vocab_size=int(metadata['num_item_classes']),
        text_embed_dim=metadata['embed_dim'],
        id_embed_dim=retrieval_config['model']['id_embed_dim'],
        hidden_dim=retrieval_config['model']['hidden_dim'],
        dropout_p=retrieval_config['model']['dropout_p'],
        lstm_hidden_size=retrieval_config['model']['lstm_hidden_size']
    ).to(device)

    state_dict = torch.load(config['retrieval']['checkpoint_path'], map_location=device)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    logger.info('Loaded retrieval model')
    return model

def load_ranking_model(config, device):
    model = RankingModel(
        embed_dim=config['ranking']['embed_dim'],
        hidden_dims=config['ranking']['hidden_dims'],
        dropout_p=config['ranking']['dropout_p']
    ).to(device)

    state_dict = torch.load(config['ranking']['checkpoint_path'], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info('Loaded ranking model')
    return model

#generating user embedding from history using the retrieval model's user tower
def encode_user_history(retrieval_model, history_indices, device):
    with torch.no_grad():
        hist_arr = np.array(history_indices, dtype=np.int64)

        hist_text = torch.from_numpy(embeddings[hist_arr]).float().to(device)
        hist_ids = torch.from_numpy(idx_map[hist_arr]).long().to(device)

        hist_text = hist_text.unsqueeze(0)
        hist_ids = hist_ids.unsqueeze(0)
        mask = torch.ones_like(hist_ids, dtype=torch.float32)

        user_emb = retrieval_model.user_tower(hist_text, mask, hist_ids)

        return user_emb

#scoring candidates using the ranking model
def rank_candidates(ranking_model, user_emb, candidate_embeds, device):
    with torch.no_grad():
        num_candidates = candidate_embeds.shape[0]

        user_emb_expanded = user_emb.expand(num_candidates, -1)

        logits = ranking_model(user_emb_expanded, candidate_embeds)

        scores = torch.sigmoid(logits)

        return scores.cpu().numpy()


#load all models and data on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting up inference API...')

    config = load_config('configs/inference_config.yaml')
    device = 'cpu'

    #load models
    state['retrieval_model'] = load_retrieval_model(config, device)
    state['ranking_model'] = load_ranking_model(config, device)
    state['device'] = device

    #load FAISS index
    logger.info('Loading FAISS index...')
    state['faiss_index'] = faiss.read_index(config['faiss']['index_path'])
    state['num_candidates'] = config['faiss']['num_candidates']
    logger.info(f'FAISS index loaded with {state["faiss_index"].ntotal} items')

    #load item embeddings
    state['item_embeds'] = np.load(config['faiss']['item_embeds_path'])
    logger.info(f'Item embeddings loaded: {state["item_embeds"].shape}')

    #load FAISS metadata 
    with open(config['faiss']['metadata_path'], 'rb') as f:
        state['faiss_metadata'] = pickle.load(f)

    #build user history lookup
    logger.info('Building user history map...')
    state['user_history'], state['title_map'] = build_user_history_map(config)
    logger.info(f'Built history for {len(state["user_history"])} users')

    state['config'] = config

    logger.info('API ready to serve requests')

    yield

    #cleanup
    state.clear()
    logger.info('API shutdown complete')


app = FastAPI(
    title='Movie Recommendation API',
    description='Two-Stage Recommendation System: Retrieval (Two-Tower) + Neural Ranking',
    version='1.0.0'
)

app.router.lifespan_context = lifespan

@app.get('/health')
def health():
    return {
        'status': 'healthy',
        'models_loaded': {
            'retrieval': state.get('retrieval_model') is not None,
            'ranking': state.get('ranking_model') is not None,
            'faiss': state.get('faiss_index') is not None
        },
        'catalog_size': state['faiss_index'].ntotal if state.get('faiss_index') else 0,
        'num_users': len(state.get('user_history', {}))
    }


@app.get('/recommend/{user_id}')
def recommend(user_id: int, top_k: int = 20, exclude_seen: bool = True):
    start_time = time.time()

    if user_id not in state['user_history']:
        raise HTTPException(status_code=404, detail=f'User {user_id} not found')

    device = state['device']
    history = state['user_history'][user_id]

    #step 1: encode user history
    user_emb = encode_user_history(state['retrieval_model'], history, device)

    #step 2: FAISS retrieval
    user_emb_np = user_emb.cpu().numpy().astype('float32')
    k = state['num_candidates']

    if exclude_seen:
        search_k = k + len(history)
    else:
        search_k = k

    scores, indices = state['faiss_index'].search(user_emb_np, search_k)
    scores = scores[0]
    indices = indices[0]

    #filter seen items
    seen_set = set(history) if exclude_seen else set()
    candidate_indices = []
    retrieval_scores = []

    for idx, score in zip(indices, scores):
        if idx == -1:
            continue
        if idx in seen_set:
            continue
        candidate_indices.append(int(idx))
        retrieval_scores.append(float(score))
        if len(candidate_indices) >= k:
            break

    #step 3: get candidate embeddings and rank
    candidate_embeds_np = state['item_embeds'][candidate_indices]
    candidate_embeds = torch.from_numpy(candidate_embeds_np.copy()).float().to(device)

    ranking_scores = rank_candidates(state['ranking_model'], user_emb, candidate_embeds, device)

    #step 4: sort by ranking score and return top-K
    sorted_order = np.argsort(ranking_scores)[::-1]

    recommendations = []
    for i, pos in enumerate(sorted_order[:top_k]):
        global_idx = candidate_indices[pos]
        meta = state['faiss_metadata'].get(global_idx, {})

        recommendations.append({
            'rank': i + 1,
            'title': meta.get('title', 'Unknown'),
            'item_id': meta.get('item_id', 'unknown'),
            'global_idx': global_idx,
            'ranking_score': round(float(ranking_scores[pos]), 4),
            'retrieval_score': round(retrieval_scores[pos], 4)
        })

    inference_time = (time.time() - start_time) * 1000  # ms

    return {
        'user_id': user_id,
        'recommendations': recommendations,
        'inference_time_ms': round(inference_time, 2),
        'retrieval_candidates': len(candidate_indices),
        'history_length': len(history)
    }


@app.get('/user/{user_id}/history')
def user_history(user_id: int):
    if user_id not in state['user_history']:
        raise HTTPException(status_code=404, detail=f'User {user_id} not found')

    history = state['user_history'][user_id]
    title_map = state['title_map']

    items = []
    for global_idx in history:
        info = title_map.get(global_idx, {})
        items.append({
            'global_idx': global_idx,
            'title': info.get('title', 'Unknown'),
            'item_id': info.get('item_id', 'unknown')
        })

    return {
        'user_id': user_id,
        'num_items': len(items),
        'history': items
    }
