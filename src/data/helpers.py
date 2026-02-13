import re
import pandas as pd

#for cleaning the categories, do not ask me how long this took
category_dict = {
    'Action': 'Action',
    'Action & Adventure': 'Action',
    'Live Action': 'Action',
    'Adventure': 'Adventure',
    'Space Adventure': 'Adventure',
    'Military & War': 'War',
    'War': 'War',
    'Westerns': 'Western',
    'Western': 'Western',
    'Gritty': 'Action',
    'Intense': 'Action', 
    'Electrifying': 'Action', 

    'Comedy': 'Comedy',
    'Comedy Central Presents': 'Comedy',
    'Comedia': 'Comedy',
    'The Comedies': 'Comedy',
    'Fun': 'Comedy',
    'Playful': 'Comedy',
    'Easygoing': 'Comedy',
    'Cheerful': 'Comedy',
    'Joyous': 'Comedy',
    
    'Drama': 'Drama',
    'Sentimental': 'Drama',
    'Touching': 'Drama',
    'Emotional': 'Drama',
    'Tragic': 'Drama',
    'The Tragedies': 'Drama',
    'Holocaust': 'Drama',
    'Sad': 'Drama',
    'Heartwarming': 'Drama',
    
    'Cerebral': 'Arthouse',
    'Contemplative': 'Arthouse',
    'Introspective': 'Arthouse',
    'Philosophical': 'Arthouse',
    'Bleak': 'Arthouse',
    'Indie & Art House': 'Arthouse',
    'Art House & International': 'Arthouse',
    'Arthouse': 'Arthouse',
    'American Masters Collection': 'Arthouse', 

    'Science Fiction': 'Sci-Fi',
    'Sci-Fi': 'Sci-Fi',
    'Sci-Fi & Fantasy': 'Sci-Fi',
    'Science Fiction & Fantasy': 'Sci-Fi',
    'Sci-Fi Action': 'Sci-Fi',
    'Sci-Fi Series & Sequels': 'Sci-Fi',
    'Aliens': 'Sci-Fi',
    'Alien Invasion': 'Sci-Fi',
    'Sci Fi Channel': 'Sci-Fi',
    'Star Wars': 'Sci-Fi',
    'Fantasy': 'Fantasy',
    'Harry Potter': 'Fantasy',
    'Fantastic': 'Fantasy',
    
    'Horror': 'Horror',
    'Horror & Suspense': 'Horror',
    'Frightening': 'Horror',
    'Harrowing': 'Horror',
    'Ominous': 'Horror',
    'Scary': 'Horror',
    'Eerie': 'Horror',
    'Terrifying': 'Horror',
    'Suspense': 'Thriller',
    'Mystery & Suspense': 'Thriller',
    'Mystery & Thrillers': 'Thriller',
    'Thrilling': 'Thriller',
    'Anxious': 'Thriller',
    'Mysterious': 'Thriller',
    'Tense': 'Thriller',
    'Dark': 'Thriller',
    
    'Kids & Family': 'Kids',
    'Kids': 'Kids',
    'Children\'s': 'Kids',
    'Cartoon Network': 'Kids',
    'Animation': 'Animation',
    'Anime': 'Anime',
    'Anime & Manga': 'Anime',
    
    'Romance': 'Romance',
    'Passionate': 'Romance',
    'Sweet': 'Romance',
    'Feel-good': 'Feel-Good',
    'Inspiring': 'Feel-Good',
    'Optimistic': 'Feel-Good',
    'Charming': 'Feel-Good',

    'Cult Movies': 'Cult Classics',
    'Cult Classics': 'Cult Classics',
    'Surreal': 'Surreal',
    'Outlandish': 'Surreal',
    'Dreamlike': 'Surreal',
    'Strange': 'Surreal',
    'Psychedelic': 'Surreal',

    'Documentary': 'Documentary',
    'Special Interests': 'Documentary',
    'Special Interest': 'Documentary',
    'Docurama': 'Documentary',
    'Educational': 'Documentary',
    'Biographies': 'Documentary',
    'History': 'Documentary',
    'Historical': 'Documentary',
    'Reality TV': 'Reality TV',
    
    'Music': 'Music',
    'Music Videos & Concerts': 'Music',
    'Opera': 'Music',
    'Music Artists': 'Music',
    'Musicals': 'Musical',
    'Musicals & Performing Arts': 'Musical',
    'Arts': 'Arts',
    'Performing Arts': 'Arts',
    'Arts, Entertainment, and Culture': 'Arts',

    'Foreign Films': 'International',
    'France': 'International',
    'French': 'International',
    'German': 'International',
    'Germany': 'International',
    'Spanish Language': 'International',
    'Spanish': 'International',
    'Japan': 'International',
    'Japanese': 'International',
    'Hong Kong': 'International',
    'Chinese': 'International',
    'Mandarin': 'International',
    'Australia & New Zealand': 'International',
    'Russia': 'International',
    'Italy': 'International',
    'Italian': 'International',
    'Bollywood': 'International',
    'Hindi': 'International',
    'Korean': 'International',
    
    'Sports': 'Sports',
    'Fitness': 'Fitness',
    'Exercise & Fitness': 'Fitness',
    'Fitness & Yoga': 'Fitness',
    'Faith': 'Faith',
    'Faith & Spirituality': 'Faith',
    'Christian Movies & TV': 'Faith',
    'Bible': 'Faith',
    'Christian Video': 'Faith',
    'Religion': 'Faith',
    'LGBTQ': 'LGBTQ',
    'Holidays & Seasonal': 'Holiday',
    'Christmas': 'Holiday',
}

#function to clean categories
def clean_categories(categories):
    #return unknown if categories are missing 
    if(not isinstance(categories, list) or len(categories)==0):
        return ['Unknown']
    
    #using a set to avoid duplicates
    cleaned_categories = set()

    #iterating over all categories in the list
    for cat in categories:

        cat = cat.strip()

        #checking and adding clean categories
        if cat in category_dict:
            cleaned_categories.add(category_dict[cat])

    if len(cleaned_categories)==0:
        #return unknown if no valid category
        return ['Unknown']
    else:
        #returning the clean list of categories
        return list(cleaned_categories)


def backfill_categories(row):
    #check if row is unknown
    if row['categories'] == ['Unknown']:
        #check if we can backfill
        if isinstance(row['details'], dict) and 'Genre' in row['details']:
            category = row['details']['Genre']
            
            #if found then return
            if isinstance(category, str):
                return [category]
            elif isinstance(category, list):
                return category
    
    #base catch all
    return row['categories']

#for converting list of strs into a str
def flatten_description(description):
    #return an empty string if description is empty or null
    if not isinstance(description, list) or len(description) == 0:
        return ''

    meaningful_parts = []

    #considering only strings that are greater than a specified length
    for s in description:
        if isinstance(s, str) and len(s.strip()) > 5:
            meaningful_parts.append(s.strip())
    
    #if we are left with nothing, better to return noise than nothing
    if not meaningful_parts:
        return ' '.join([str(s) for s in description if s is not None])

    return ' '.join(meaningful_parts)

#for removing invalid chars
def sanitize_text(text):
    if not isinstance(text, str) or len(text) == 0:
        return ''    

    #removing see more
    txt = re.sub(r'see more', '', text, flags=re.IGNORECASE)
    
    #replacing random noise
    txt = txt.replace('&nbsp;', ' ').replace('nbsp', ' ')

    #replacing html tags
    txt = re.sub(r'<[^>]+>', ' ', txt)

    #removing backslashes
    txt = txt.replace("\\", "")
    
    #keeping only alphanumeric, punctuation and space chars
    txt = re.sub(r'[^a-zA-Z0-9\s.;,:!?"\'/%$-]+', '', txt)
    
    #removing trailing and preceeding spaces
    txt = re.sub(r'\s+', ' ', txt).strip()

    return txt

key_mapping = {
    'Run time': 'Duration',
    'Runtime': 'Duration',
    'Directors': 'Director',
    'Director': 'Director',
    'Actors': 'Cast',
    'Starring': 'Cast',
    'Media Format': 'Format',
    'Format': 'Format',
    'Audio languages': 'Language',
    'Dubbed': 'Language',
    'Subtitles': 'Subtitles',
    'Release date': 'Release',
    'Publication Date': 'Release',
}

allowed = ['Cast', 'Director','Studio', 'Format', 'Language', 'Subtitles', 'Rating', 'Duration', 'Release']

def clean_details(details):
    #return an empty string if the dict is empty
    if not isinstance(details, dict):
        return ''
    #a dict for select features
    extracted_features = {}

    #checking for valid keys to add
    for key, item in details.items():
        key = key.strip()
        if key in key_mapping.keys():
            final = key_mapping[key]
        elif key in allowed:
            final = key
        else:
            continue
        
        #handling lists
        if isinstance(item, list):
            value = ', '.join([str(s) for s in item])
        else:
            value = str(item)

        #year matters more than day and month
        if final=='Release':
            value = pd.to_datetime(value, errors='coerce')
            value = value.year
            final = 'Year'
        
        #only adding one if collision exists
        if final not in extracted_features:
            extracted_features[final] = value
    
    text_parts = [f"{k}: {v}" for k, v in extracted_features.items()]
    
    return "; ".join(text_parts)

#function for creating the column that will serve to produce the column we will use for understanding the product
def create_unified_text(row):
    title_text = f'Title: {row['title'].strip()}'

    genre_str = ', '.join(row['categories'])
    genre_text = f'Genre: {genre_str}'

    #only add details if they exist
    if row['details']:
        details_text = f'Details: {row['details'].strip()}'
    else:
        details_text = ''

    #only add a description if it exists
    if row['description'].strip():
        desc_text = f'Description: {row['description'].strip()}'
    else:
        desc_text = ''

    #concatenate
    parts = [title_text, genre_text, details_text, desc_text]
    
    parts = [p for p in parts if p]

    #because we will use bert to generate embeddings
    return ' [SEP] '.join(parts)

#for creating training samples with history, target item, and target rating
def generate_sliding_window(df, group_col, asin_col, rating_col, max_len=100):
    samples = []

    for user_id, user_df in df.groupby(group_col):

        user_df = user_df.sort_values('timestamp')

        asins = user_df[asin_col].tolist()
        ratings = user_df[rating_col].tolist()
        item_idxs = user_df['item_idx'].tolist()

        if len(asins) < 2:
            continue

        for i in range(1, len(asins)):
            start = max(0, i - max_len)

            samples.append({
                'user_idx': user_id,
                'history_asins': asins[start:i],
                'target_parent_asin': asins[i],
                'target_item_idx': item_idxs[i],
                'target_rating': ratings[i]
            })

    return pd.DataFrame(samples)
