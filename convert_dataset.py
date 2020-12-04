import json, pickle
from transformers import GPT2Tokenizer

datafile_json = 'personachat_self_original.json'
datafile_pickle = 'personachat_self_cached.pickle'
datafile_personalities = 'personachat_self_personalities.pickle'

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize(obj):
    if isinstance(obj, list):
        return [tokenize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: tokenize(v) for k, v in obj.items()}
    return tokenizer.encode(obj)

# Load the data file
with open(datafile_json) as fp:
    dataset = json.load(fp)

# Tokenize it recursively
dataset = tokenize(dataset)

# Save it
with open(datafile_pickle, 'wb') as fp:
    pickle.dump(dataset, fp)

# Extract all personalities from the training dataset
personalities = [dialog['personality'] for dialog in dataset['train']]

# And save it
with open(datafile_personalities, 'wb') as fp:
    pickle.dump(personalities, fp)
