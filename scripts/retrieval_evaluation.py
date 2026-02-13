from src.models.two_tower import TwoTowerModel
from src.data.retrieval_dataset import RetrievalEvalDataset, eval_collate_fn, embeddings, idx_map
from src.evaluation.retrieval_evaluator import TestEvaluator
from src.utils.logging import get_logger
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
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
    
    #loading model
    model = TwoTowerModel(
        id_vocab_size=int(metadata['num_item_classes']),
        text_embed_dim=metadata['embed_dim'],
        id_embed_dim=config['model']['id_embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout_p=config['model']['dropout_p'],
        lstm_hidden_size=config['model']['lstm_hidden_size']
    ).to(device)

    #loading checkpoint
    checkpoint_path = args.checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    #because this model used torch.compile()
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        logger.info('Detected torch.compile() checkpoint, stripped _orig_mod. prefix')
    
    model.load_state_dict(state_dict)
    
    logger.info(f'Loaded model from {checkpoint_path}')
    
    #loading test data
    test_df = pd.read_parquet(args.test_path)
    test_dataset = RetrievalEvalDataset(test_df, embeddings, idx_map)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['eval_batch_size'], shuffle=False, collate_fn=eval_collate_fn, num_workers=0)
    
    logger.info(f'Loaded test set with {len(test_dataset)} samples')
    
    #computing item popularity from training data for novelty metric
    train_df = pd.read_parquet(config['data']['train_path'])
    item_counts = train_df['target_global_idx'].value_counts()
    total_interactions = len(train_df)
    item_popularity = (item_counts / total_interactions).to_dict()
    
    logger.info('Computed item popularity for novelty metric')
    
    #running evaluation
    evaluator = TestEvaluator(model, embeddings, idx_map)
    results = evaluator.evaluate(test_loader, item_popularity)
    
    #printing results
    logger.info('Test Set Evaluation Results')
    
    logger.info('\nAccuracy Metrics')
    logger.info(f"MRR: {results['MRR']:.4f}")
    for k in [10, 20, 50, 100, 500, 1000, 1500, 2000]:
        logger.info(f"Recall@{k}: {results[f'Recall@{k}']:.4f} ({results[f'Recall@{k}']*100:.2f}%)")
        logger.info(f"NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
    
    logger.info('\nCoverage Metrics')
    logger.info(f"Catalog Coverage: {results['Catalog_Coverage']:.4f} ({results['Catalog_Coverage']*100:.2f}%)")
    logger.info(f"Unique Items Recommended: {results['Unique_Items_Recommended']:,} / {results['Total_Items_in_Catalog']:,}")
    
    logger.info('\nDiversity Metrics')
    for k in [10, 20, 50]:
        if f'Diversity@{k}' in results:
            logger.info(f"Diversity@{k}: {results[f'Diversity@{k}']:.4f}")
    
    logger.info('\nNovelty Metrics')
    for k in [10, 20, 50, 100, 500, 1000, 1500, 2000]:
        if f'Novelty@{k}' in results:
            logger.info(f"Novelty@{k}: {results[f'Novelty@{k}']:.4f}")
        
    #save results to file
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Results saved to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate retrieval model on test set')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='../data/processed/test.parquet', help='Path to test parquet')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    
    args = parser.parse_args()
    main(args)
