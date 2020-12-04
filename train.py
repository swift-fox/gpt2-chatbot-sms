import pickle
import tensorflow as tf

from itertools import chain
from transformers import AutoTokenizer, TFAutoModelForCausalLM

datafile = 'personachat_self_cached.pickle'
model_path = 'personachat-distilgpt2'

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = TFAutoModelForCausalLM.from_pretrained('distilgpt2', return_dict=False)

# Add special tokens to mark different participants
tokenizer.add_special_tokens({'pad_token': '<pad>', 'additional_special_tokens': ['<bot>','<person>']})

with tf.name_scope('tfgp_t2lm_head_model/transformer/wte'): # Workaround for a bug in the transformers library
    model.resize_token_embeddings(len(tokenizer))

bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
bot, person = tokenizer.encode('<bot>', '<person>')

# Data pipeline
input_shape = {'input_ids': [None], 'token_type_ids': [None], 'labels': [None]}
input_type = {'input_ids': tf.int32, 'token_type_ids': tf.int32, 'labels': tf.int32}
input_padding = {'input_ids': pad, 'token_type_ids': pad, 'labels': -100}

def prepare(dataset):
    def gen():
        for dialog in dataset:
            personality = [bos] + list(chain(*dialog['personality']))
            for utterance in dialog['utterances']:
                history = [[bot if i % 2 else person] + s for i, s in enumerate(utterance['history'])]
                reply = [bot] + utterance['candidates'][-1] + [eos] # Use only the ground truth

                seq = [personality] + history + [reply]
                input_ids = list(chain(*seq))
                token_type_ids = [i % 2 for i, s in enumerate(seq) for _ in s]
                labels = [-100] * (len(input_ids) - len(reply) + 1) + reply[1:]

                yield {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'labels': labels}

    return tf.data.Dataset.from_generator(gen, input_type, input_shape)

# Load the cached data file
with open(datafile, 'rb') as fp:
    dataset = pickle.load(fp)

train = prepare(dataset['train']).shuffle(100).padded_batch(2, input_shape, input_padding)
valid = prepare(dataset['valid']).padded_batch(2, input_shape, input_padding)

# Train it!
optimizer = tf.keras.optimizers.Adam(6e-5)
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')

for epoch in range(3):
    for batch in train:
        with tf.GradientTape() as tape:
            loss, logits, _ = model(batch, training=True)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)    # Record training loss for statistics

    for batch in valid:
        loss, logits, _ = model(batch)
        valid_loss(loss)    # Record validation loss for statistics

    print("epoch {}, train_loss: {:.4f}, valid_loss: {:.4f}".format(epoch + 1, train_loss.result(), valid_loss.result()))
    train_loss.reset_states()
    valid_loss.reset_states()

# Save the model
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
