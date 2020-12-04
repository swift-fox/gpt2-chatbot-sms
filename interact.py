import pickle, random
import tensorflow as tf

from itertools import chain
from transformers import AutoTokenizer, TFAutoModelForCausalLM

datafile = 'personachat_self_personalities.pickle'
model_path = 'personachat-distilgpt2'

N = 20  # Maximum number of words to output
k = 10   # Top-k items to select from
p = 0.8 # Top-p cumulative probability to select from

# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForCausalLM.from_pretrained(model_path, use_cache=True, return_dict=False)

bos = tokenizer.bos_token_id
bot, person = tokenizer.encode('<bot>', '<person>')

# Load personalities dataset
with open(datafile, 'rb') as fp:
    personalities = pickle.load(fp)

def feed(input_ids, speaker, past=None):
    return model({
        'input_ids': tf.constant(input_ids),
        'token_type_ids': tf.fill(len(input_ids), int(speaker == person))
    }, past)

def new_chat():
    # Select a random personality from the dataset to feed the model
    personality = random.choice(personalities)
    return feed([bos] + list(chain(*personality)), bot)

def generate_reply(input_string, past):
    _, past = feed([person] + tokenizer.encode(input_string), person, past)

    input_ids = tf.constant([bot])
    output = []
    for _ in range(N):
        logits, past = model({'input_ids': input_ids, 'token_type_ids': tf.constant([0])}, past)

        # Top-k filtering: select only k items with largest probabilities
        logits, indices = tf.math.top_k(logits[0], k)

        # Top-p filtering: select only top items within cumulative probability of p
        cumsum = tf.math.cumsum(tf.nn.softmax(logits))
        selected = cumsum <= max(p, cumsum[0])   # Make sure at least 1 item is selected
        logits, indices = logits[selected], indices[selected]

        # Reconstruct the logits with values and indices
        _inf = tf.fill([len(tokenizer)], -float('inf'))
        logits = tf.tensor_scatter_nd_update(_inf, indices[:,tf.newaxis], logits)

        # Sample the next word
        input_ids = tf.random.categorical(logits[tf.newaxis,:], 1)[0]
        if input_ids[0].numpy() in tokenizer.all_special_ids:
            break

        output.append(input_ids[0].numpy())

    return tokenizer.decode(output), past

if __name__ == "__main__":
    _, past = new_chat()

    print('==== Conversation starts ====')
    while True:
        input_string = input('You: ')
        output, past = generate_reply(input_string, past)
        print('Bot: ' + output)
