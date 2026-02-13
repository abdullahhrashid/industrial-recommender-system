import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.helpers import generate_sliding_window
from src.utils.logging import get_logger
import joblib
import os

#intialzing logger
logger = get_logger(__file__)

#paths to data
item_path = os.path.join(os.path.dirname(__file__), '../../data/interim/products.csv')
review_path = os.path.join(os.path.dirname(__file__), '../../data/interim/interactions.csv')

#reading the data
item_df = pd.read_csv(item_path)
review_df = pd.read_csv(review_path)

logger.info('Loaded datasets')

#creating a set of valid items that exist in the processed metadata
valid_items = set(item_df['parent_asin'])

#filtering the reviews to only keep rows where the item exists in metadata
review_df = review_df[review_df['parent_asin'].isin(valid_items)]

#count interactions per user
user_counts = review_df['user_id'].value_counts()

#keeping only users with >= 5 interactions
valid_users = user_counts[user_counts >= 5].index
review_df = review_df[review_df['user_id'].isin(valid_users)]

#we will use this in the loss function only
review_df['rating'] = review_df['rating'] / 5.0

#after some contemplation, it is not worth it to keep these two
item_df = item_df.drop(columns=['average_rating', 'rating_number'])

#creating integer user ids
user_encoder = LabelEncoder()
review_df['user_idx'] = user_encoder.fit_transform(review_df['user_id'])

#just making sure
review_df['timestamp'] = pd.to_datetime(review_df['timestamp'])

#aiming for a user level temporal split
review_df = review_df.sort_values(['user_idx', 'timestamp'])

#splitting into test, train and validation
test = review_df.groupby('user_idx').tail(1)
review_df = review_df.drop(test.index)
val = review_df.groupby('user_idx').tail(1)
train = review_df.drop(val.index)

#creating integer product ids
item_encoder = LabelEncoder()
item_encoder.fit(train['parent_asin']) 

logger.info('Split the data into train, validation and test sets by leveraging a user level temporal split. Created User and Item IDs')

train = train.copy()
test = test.copy()
val = val.copy()
item_df = item_df.copy()

train['item_idx'] = item_encoder.transform(train['parent_asin']) + 1

#we will use this to convert our asin to indices
warm_item_map = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_) + 1))

#mapping ids to indices
val['item_idx'] = val['parent_asin'].map(warm_item_map).fillna(0).astype(int)
test['item_idx'] = test['parent_asin'].map(warm_item_map).fillna(0).astype(int)
item_df['item_idx'] = item_df['parent_asin'].map(warm_item_map).fillna(0).astype(int)

#generating training samples with user history
train_samples = generate_sliding_window(train, 'user_idx', 'parent_asin', 'rating')

base_history_map = train.sort_values('timestamp').groupby('user_idx')['parent_asin'].apply(list).to_dict()

#to avoid really long history sequences
MAX_LEN = 100

val['history_asins'] = val['user_idx'].map(lambda u: base_history_map.get(u, [])[-MAX_LEN:])

def get_test_history(user_id):
    hist = base_history_map.get(user_id, [])
    val_item = val.loc[val['user_idx'] == user_id, 'parent_asin']

    if len(val_item) > 0:
        hist = hist + [val_item.iloc[0]]

    return hist[-MAX_LEN:]

test['history_asins'] = test['user_idx'].apply(get_test_history)

val['target_item_idx'] = val['item_idx']
val['target_rating'] = val['rating']
val['target_parent_asin'] = val['parent_asin']
   
test['target_item_idx'] = test['item_idx']
test['target_rating'] = test['rating']
test['target_parent_asin'] = test['parent_asin']

logger.info('Created User History and engineered some other features')

#dropping unneeded columns
val = val.drop(columns=['user_id', 'timestamp','rating', 'item_idx', 'parent_asin'])
test = test.drop(columns=['user_id', 'timestamp','rating', 'item_idx', 'parent_asin'])

#paths to store encoders
item_enc_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/item_encoder.joblib')
user_path = os.path.join(os.path.dirname(__file__), '../../data/artifacts/user_encoder.joblib')

#paths to store datasets
train_path = os.path.join(os.path.dirname(__file__), '../../data/interim/train.parquet')
val_path = os.path.join(os.path.dirname(__file__), '../../data/interim/val.parquet')
test_path = os.path.join(os.path.dirname(__file__), '../../data/interim/test.parquet')
out_item_path = os.path.join(os.path.dirname(__file__), '../../data/interim/items.csv')

#parquet because it's better for lists
train_samples.to_parquet(train_path, index=False)
val.to_parquet(val_path, index=False)
test.to_parquet(test_path, index=False)
item_df.to_csv(out_item_path, index=False)

#storing encoders so that we can use them to decode user id and asin
joblib.dump(item_encoder, item_enc_path)
joblib.dump(user_encoder, user_path)

logger.info('Saved data and artifacts')
