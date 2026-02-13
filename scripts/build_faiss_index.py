import numpy as np
import faiss
import argparse
from src.utils.logging import get_logger

logger = get_logger(__file__)

def main(args):
    #loading item embeddings
    logger.info(f'Loading item embeddings from {args.embeddings_path}')
    item_embeddings = np.load(args.embeddings_path)
    
    num_items, embed_dim = item_embeddings.shape
    logger.info(f'Loaded {num_items} items with dimension {embed_dim}')
    
    #converting to float32 for faiss
    item_embeddings = item_embeddings.astype('float32')
    
    #im using a flat index because my number of items is less than a million
    logger.info('Building IndexFlatIP (exact search)')
    index = faiss.IndexFlatIP(embed_dim)
    index.add(item_embeddings)
    
    logger.info(f'Index built successfully with {index.ntotal} vectors')
    
    #saving the faiss index
    logger.info(f'Saving index to {args.output_path}')
    faiss.write_index(index, args.output_path)
    
    logger.info('FAISS index build complete')
    
    #quick sanity check
    logger.info('Running sanity check...')
    test_query = item_embeddings[0:1]
    distances, indices = index.search(test_query, 5)
    logger.info(f'Test query returned indices: {indices[0]}')
    logger.info(f'Test query returned scores: {distances[0]}')
    logger.info('Sanity check passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build FAISS index from item embeddings')
    
    parser.add_argument('--embeddings_path', type=str, default='../data/artifacts/faiss_item_embeddings.npy', help='Path to item embeddings .npy file')
    parser.add_argument('--output_path', type=str, default='../data/artifacts/faiss_index.bin', help='Path to save FAISS index')
    
    args = parser.parse_args()
    main(args)
