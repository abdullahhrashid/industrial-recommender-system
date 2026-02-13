import pandas as pd
from helpers import clean_categories, flatten_description, sanitize_text, backfill_categories, clean_details, create_unified_text
import os
from src.utils.logging import get_logger

#setting up logger
logger = get_logger(__file__)

#path to data
path = os.path.join(os.path.dirname(__file__), '../../data/raw/meta_Movies_and_TV.jsonl.gz')
    
#reading the data
df = pd.read_json(path, lines=True)

#dropping unneccesary columns
df = df.drop(columns=['subtitle', 'author', 'bought_together', 'images', 'videos', 'store', 'main_category', 'price', 'features'])

#cleaning categories
df['categories'] = df['categories'].apply(clean_categories)

#backfilling from details if possible
df['categories'] = df.apply(backfill_categories, axis=1)

df = df.dropna(subset=['title', 'parent_asin', 'average_rating', 'rating_number']).reset_index(drop=True)

#converting lists into strings
df['description'] = df['description'].apply(flatten_description)

#removing invalid chars
df['description'] = df['description'].apply(sanitize_text)

#converting dicts into strings
df['details'] = df['details'].apply(clean_details)

#removing invalid chars
df['details'] = df['details'].apply(sanitize_text)

logger.info('Loaded product data and cleaned columns.')

#creating the main column that helps the model understands the product
df['input_text'] = df.apply(create_unified_text, axis=1)

#we won't be needing these anymore
df = df.drop(columns = ['categories', 'details', 'description'])

logger.info('Created the unified input text column and dropped unneeded columns.')

path = os.path.join(os.path.dirname(__file__), '../../data/interim/products.csv')

#saving as a csv file
df.to_csv(os.path.join(path), index=False)

logger.info('Saved Product Data')
