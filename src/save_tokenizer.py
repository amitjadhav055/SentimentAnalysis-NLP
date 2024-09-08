import json
from tensorflow.keras.preprocessing.text import Tokenizer
import os

# Assuming you have already created and fitted your tokenizer
# For demonstration, we'll create a new tokenizer and fit it with some sample data
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')

# Sample data (replace this with your actual data)
sample_texts = [
    "This is a great product!",
    "I didn't like the product.",
    "Best purchase I've made this year.",
    "The product was okay, not great.",
    "Wouldn't recommend it."
]
tokenizer.fit_on_texts(sample_texts)

# Save tokenizer as JSON
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/tokenizer.pkl', 'w') as file:
    json.dump(tokenizer.to_json(), file)

print("Tokenizer saved successfully as JSON!")
