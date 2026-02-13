import numpy as np
import faiss
import argparse
from tqdm import tqdm
from src.utils.logging import get_logger

logger = get_logger(__file__)

def main(args):
    logger.info(f'Loading FAISS index from {args.index_path}')
    index = faiss.read_index(args.index_path)
    logger.info(f'Index has {index.ntotal} items')

    for split in ['train', 'val', 'test']:
        embeds_path = f'../data/artifacts/{split}_user_embeds.npy'

        logger.info(f'Loading {split} user embeddings from {embeds_path}')
        user_embeds = np.load(embeds_path).astype('float32')

        num_users = len(user_embeds)
        logger.info(f'{split}: {num_users} interactions, retrieving top {args.k} candidates each')

        #batch FAISS search for efficiency
        batch_size = args.batch_size
        all_indices = np.zeros((num_users, args.k), dtype=np.int32)

        for start in tqdm(range(0, num_users, batch_size), desc=f'FAISS search ({split})'):
            end = min(start + batch_size, num_users)
            batch = user_embeds[start:end]

            _, indices = index.search(batch, args.k)
            all_indices[start:end] = indices.astype(np.int32)

        output_path = f'../data/artifacts/{split}_candidates.npy'
        np.save(output_path, all_indices)
        logger.info(f'Saved {split} candidates to {output_path} (shape: {all_indices.shape})')

    logger.info('Candidate pre-computation complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute FAISS candidates for ranking')
    parser.add_argument('--index_path', type=str, default='../data/artifacts/faiss_index.bin', help='Path to FAISS index')
    parser.add_argument('--k', type=int, default=2000, help='Number of candidates to retrieve')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for FAISS search')

    args = parser.parse_args()
    main(args)
