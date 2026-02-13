from src.models.ranking import RankingModel
from src.data.ranking_dataset import RankingEvalDataset, ranking_eval_collate_fn
from src.evaluation.ranking_evaluator import RankingEvaluator
from src.utils.logging import get_logger
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

logger = get_logger(__file__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    config = load_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Running on {device}')

    #loading pre-computed embeddings
    item_embeds = np.load(config['data']['item_embeds_path'])
    test_user_embeds = np.load('../data/artifacts/test_user_embeds.npy')
    test_candidates = np.load('../data/artifacts/test_candidates.npy')

    logger.info(f'Loaded item embeddings: {item_embeds.shape}')
    logger.info(f'Loaded test user embeddings: {test_user_embeds.shape}')
    logger.info(f'Loaded test candidates: {test_candidates.shape}')

    #loading test interactions
    test_df = pd.read_parquet('../data/processed/test.parquet')
    logger.info(f'Loaded test data: {len(test_df)} interactions')

    #creating dataset
    test_dataset = RankingEvalDataset(
        user_embeds=test_user_embeds,
        interactions=test_df,
        item_embeds=item_embeds,
        candidates=test_candidates
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['eval_batch_size'],
        shuffle=False,
        collate_fn=ranking_eval_collate_fn,
        num_workers=0
    )

    #loading trained ranking model
    model = RankingModel(
        embed_dim=config['model']['embed_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_p=config['model']['dropout_p']
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f'Loaded ranking model from {args.checkpoint}')

    #evaluating
    evaluator = RankingEvaluator(model=model, device=device)
    metrics = evaluator.evaluate(test_loader)

    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logger.info(f'Test Metrics: {metrics_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Ranking Model on Test Set')
    parser.add_argument('--config', type=str, required=True, help='Path to ranking config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained ranking model checkpoint')

    args = parser.parse_args()
    main(args)
