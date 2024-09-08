import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset with column names
df = pd.read_csv('data/amazon_reviews_train.csv', names=['review', 'sentiment'], header=None)

# Vectorized preprocessing function with .loc[] for assignment
def preprocess_text_vectorized(df):
    # Remove rows with missing values in 'review'
    df = df.dropna(subset=['review']).copy()  # Ensure we work on a copy

    # Convert reviews to string type, just in case there are any non-string values
    df.loc[:, 'review'] = df['review'].astype(str)
    
    # Convert reviews to lowercase
    df.loc[:, 'review'] = df['review'].str.lower()
    
    # Tokenize the reviews
    df.loc[:, 'review'] = df['review'].apply(word_tokenize)
    
    # Remove stopwords
    df.loc[:, 'cleaned_review'] = df['review'].apply(
        lambda tokens: " ".join([word for word in tokens if word not in stopwords.words('english')])
    )
    
    return df

# Apply the vectorized preprocessing
df = preprocess_text_vectorized(df)

# Save the cleaned data
df.to_csv('data/cleaned_reviews.csv', index=False)

print("Data preprocessing completed and saved to 'data/cleaned_reviews.csv'")
