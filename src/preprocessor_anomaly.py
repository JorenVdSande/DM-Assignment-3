import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Load the spaCy model
# disable unnecessary components to make it faster since we only need lemmas/tags
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

HTML_TAGS = [
    "html","head","body","div","span","p","a","img","br","hr",
    "ul","ol","li","table","tr","td","th","thead","tbody",
    "h1","h2","h3","h4","h5","h6","strong","em","b","i","u",
    "form","input","button","label","script","style","link","meta"
]

def normalize_urls(text):
    """
    Splits URLs into meaningful tokens without affecting non-URL text.
    """
    def _process_url(match):
        url = match.group(0)

        # Remove protocol (http://, https://)
        url = re.sub(r'^https?://', 'http: ', url)

        # Replace separators with spaces
        url = re.sub(r'[\/\-\_\?\=\&]', ' ', url)

        # Remove dots except between domain parts → keep them meaningful
        url = re.sub(r'\.(?=\s|$)', ' ', url)  # trailing dots
        url = url.replace('.', ' ')  # split domain parts

        return url

    # Only match real URLs
    return re.sub(r'https?://\S+|www\.\S+', _process_url, text)

def clean_raw_text(text):
    """
    Initial 'brute force' cleaning before spaCy processing.
    """
    if not isinstance(text, str):
        return ""
        
    # normalize URLs
    #text = normalize_urls(text)
    
    # remove HTML tags
    #pattern = r'</?(?:' + '|'.join(HTML_TAGS) + r')[^>]*>'
    #text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # remove Email addresses
    #text = re.sub(r'\S+@\S+', '', text)
    
    # remove newlines and extra spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def spacy_preprocess(text):
    """
    Uses spaCy for tokenization, stopword removal, and lemmatization.
    """
    doc = nlp(text)
    
    # Process tokens:
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        
        if not token.is_stop and len(lemma) > 1 and any(c.isalnum() for c in lemma):
            tokens.append(lemma)
    
    return " ".join(tokens)

def remove_frequent_and_infrequent_tokens(column_data):
    documents = column_data.tolist()
    
    vectorizer = TfidfVectorizer(
        # Ignore terms that appear in more than x% of the documents
        max_df=0.8,
        
        # Ignore terms that appear in fewer than x documents
        min_df=20,
        
        ngram_range=(1, 3)
    )
    
    X = vectorizer.fit_transform(documents)
    
    # Get lists of surviving tokens
    surviving_tokens_per_doc = vectorizer.inverse_transform(X)
    
    # Transform: "space shuttle" -> "space_shuttle"
    # Then join tokens back into a single document string
    filtered_docs = []
    for tokens in surviving_tokens_per_doc:
        # Replace internal spaces with underscores for each token
        processed_tokens = [t.replace(" ", "_") for t in tokens]
        filtered_docs.append(" ".join(processed_tokens))
    
    return X, filtered_docs



def read_dataset(use_bert=False):
    """
    Loads, cleans, and vectorizes the article dataset.

    The function performs text cleaning and spaCy-based lemmatization. It generates
    a 'token_text' version of the documents for human inspection. Depending on the
    'use_bert' flag, it returns either high-dimensional BERT embeddings or a
    frequency-based TF-IDF matrix for use in downstream clustering or anomaly detection.

    Parameters:
    use_bert : bool, optional
        If True, uses 'all-roberta-large-v1' to generate semantic embeddings from
        the clean text. If False, defaults to the TF-IDF matrix.

    Returns:
    df : pd.DataFrame
        The processed dataframe containing doc_id, original text, and filtered tokens.
    X : ndarray or sparse matrix
        The feature matrix (Dense BERT embeddings or Sparse TF-IDF matrix).
    """
    
    df = pd.read_csv("data/articles.csv")
    
    # apply cleaning and preprocessing
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_raw_text)
    
    print("Running spaCy preprocessing (Lemmatization)...")
    df['spacy_processed_text'] = df['clean_text'].apply(spacy_preprocess)
    
    # Always run this to get the 'token_text' for inspection CSV
    # ignore 'X_tfidf' when using BERT
    X_tfidf, df['token_text'] = remove_frequent_and_infrequent_tokens(df['spacy_processed_text'])
    
    if use_bert:
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer('all-roberta-large-v1')
        
        print("Generating BERT embeddings...")
        X = model.encode(df['clean_text'].tolist(), show_progress_bar=True)
    else:
        X = X_tfidf
    
    # Preview the results
    print(df[['doc_id', 'token_text']].head())
    
    # Export for review
    df = df.drop(["clean_text", "spacy_processed_text"], axis=1)
    df.to_csv('data/articles_preprocessed4.csv', index=False)
    
    return df, X