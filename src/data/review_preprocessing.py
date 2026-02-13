import pandas as pd
import os
from src.utils.logging import get_logger

#setting up logger
logger = get_logger(__file__)

#path to data
path = os.path.join(os.path.dirname(__file__), '../../data/raw/Movies_and_TV.jsonl.gz')

df_list = []

#because my ram cannot handle 17.3 million rows at once, we will use lazy loading
with pd.read_json(path, lines=True, chunksize=1000000) as reader:
    for chunk in reader:
        #keeping only reviews from the year 2015 onwards, "the streaming era"
        chunk = chunk[chunk['timestamp'].dt.year>2014].copy()
        
        #dropping invalid reviews or reviews for products that users didn't like, we don't want our system recommending products that users don't like
        chunk = chunk[(chunk['rating']>=3) & (chunk['verified_purchase'])]
        
        #dropping unneeded columns
        chunk = chunk.drop(columns=['images', 'helpful_vote', 'title', 'text', 'asin', 'verified_purchase'])
        
        #dropping duplicate reviews
        chunk = chunk.drop_duplicates(keep='first')

        df_list.append(chunk)

logger.info('Loaded the Reviews dataset. Applied temporal filtering and kept only positive interactions')

#obtaining our full dataset
df = pd.concat(df_list)

#doing k core filtering to remove noise
done = False
        
while(not done):
    prev_size = df.shape[0]
        
    #removing invalid users
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts>=5].index
    df=df[df['user_id'].isin(valid_users)]
        
    #removing invalid items
    item_counts = df['parent_asin'].value_counts()
    valid_items = item_counts[item_counts>=5].index
    df=df[df['parent_asin'].isin(valid_items)]
        
    #check if we're done
    if(df.shape[0]==prev_size):
        done = True

path = os.path.join(os.path.dirname(__file__), '../../data/interim/interactions.csv')

#saving as a csv file
df.to_csv(os.path.join(path), index=False)

logger.info('Did K core filtering with a value of 5. Saved Interactions Data')
