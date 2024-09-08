import pandas as pd
import numpy as np
import tensorflow as tf

# Function to convert sentiment labels
def convert_sentiment_labels(sentiment):
    # Implement your sentiment conversion logic here
    # This is an example; replace it with actual conversion logic
    if pd.isna(sentiment):
        return np.nan
    sentiment = sentiment.strip().lower()
    if 'good' in sentiment or 'excellent' in sentiment:
        return 'positive'
    elif 'bad' in sentiment or 'poor' in sentiment:
        return 'negative'
    else:
        return 'neutral'

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("First few rows of the dataframe:")
    print(df.head())
    return df

# Preprocess data
def preprocess_data(df):
    print("Unique sentiment values before conversion:", df['sentiment'].unique())
    
    df['sentiment'] = df['sentiment'].apply(convert_sentiment_labels)

    # Debugging: Check unique sentiment values after conversion
    print("Unique sentiment values after conversion:", df['sentiment'].unique())
    
    # Debugging: Print rows with NaN in 'sentiment'
    print("Rows with NaN in 'sentiment':")
    print(df[df['sentiment'].isna()])
    
    if df['sentiment'].isna().all():
        raise ValueError("The dataframe is empty after preprocessing. Ensure that there are valid reviews and sentiment labels.")
    
    return df

# Example function for evaluating model (replace with your actual evaluation logic)
def evaluate_model(df):
    # Example placeholder for model evaluation
    print("Evaluating model...")
    # This is where you would load your trained model and evaluate it
    # For example, load model:
    # model = tf.keras.models.load_model('your_model.h5')
    # Perform evaluation, e.g.:
    # results = model.evaluate(df['cleaned_review'], df['sentiment'])
    # print(results)
    print("Model evaluation complete.")

# Main function to run the pipeline
def main():
    # Load your dataset
    df = load_data('data/cleaned_reviews.csv')

    # Preprocess data
    df = preprocess_data(df)

    # Save or process the cleaned data
    df.to_csv('processed_dataset.csv', index=False)

    # Evaluate model
    evaluate_model(df)

if __name__ == "__main__":
    main()
