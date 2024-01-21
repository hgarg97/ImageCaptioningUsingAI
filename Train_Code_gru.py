#%%
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from PIL import Image
from cache import Cache
from pickle import dump
import pickle
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import coco
coco.set_data_dir("data\\coco\\")
_, filenames_train, captions_train = coco._load_records(train=True)
num_images_train = len(filenames_train)
num_images_train
_, filenames_val, captions_val = coco._load_records(train=False)
num_images_val = len(filenames_val)
num_images_val

def load_image(path, size=None):
    img = Image.open(path)
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.array(img)
    img = img / 255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img

def show_image(idx, train):
    if train:
        dir = coco.train_dir
        filename = filenames_train[idx]
        captions = captions_train[idx]
    else:
        dir = coco.val_dir
        filename = filenames_val[idx]
        captions = captions_val[idx]
    path = os.path.join(dir, filename)
    for caption in captions:
        print(caption)
    img = load_image(path)
    plt.imshow(img)
    plt.show()

show_image(idx=1, train=True)
image_model = VGG16(include_top=True, weights='imagenet')

image_model.summary()

transfer_layer = image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)

img_size = K.int_shape(image_model.input)[1:3]
img_size

transfer_values_size = K.int_shape(transfer_layer.output)[1]
transfer_values_size

def print_progress(count, max_count):
    pct_complete = count / max_count
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def process_images(data_dir, filenames, batch_size=32):
    num_images = len(filenames)
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)
    start_index = 0
    while start_index < num_images:
        print_progress(count=start_index, max_count=num_images)
        end_index = start_index + batch_size
        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - start_index
        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(data_dir, filename)
            img = load_image(path, size=img_size)
            image_batch[i] = img
        transfer_values_batch = image_model_transfer.predict(image_batch[0:current_batch_size])
        transfer_values[start_index:end_index] = transfer_values_batch[0:current_batch_size]
        start_index = end_index
    print()
    return transfer_values

#transfer_values_train = process_images('data\\coco\\train2017', filenames_train, 32)
transfer_values_train = open('data\\coco\\transfer_values_train.pkl','rb')
transfer_values_train = pickle.load(transfer_values_train)
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)
#dump(transfer_values_train,open('data\\coco\\transfer_values_train.pkl','wb'))

#transfer_values_val = process_images('data\\coco\\val2017', filenames_val, 32)
transfer_values_val = open('data\\coco\\transfer_values_val.pkl','rb')
transfer_values_val = pickle.load(transfer_values_val)
print("dtype:", transfer_values_val.dtype)
print("shape:", transfer_values_val.shape)
#dump(transfer_values_val,open('data\\coco\\transfer_values_val.pkl','wb'))


mark_start = 'ssss '
mark_end = ' eeee'

def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end for caption in captions_list] for captions_list in captions_listlist]
    return captions_marked

captions_train_marked = mark_captions(captions_train)
captions_train_marked[0]

captions_train[0]

def flatten(captions_listlist):
    captions_list = [caption for captions_list in captions_listlist for caption in captions_list]
    return captions_list

captions_train_flat = flatten(captions_train_marked)

num_words = 10000

class TokenizerWrap(Tokenizer):
    def __init__(self, texts, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = " ".join(words)
        return text
    
    def captions_to_tokens(self, captions_listlist):
        tokens = [self.texts_to_sequences(captions_list) for captions_list in captions_listlist]
        return tokens


tokenizer = TokenizerWrap(texts=captions_train_flat, num_words=num_words)
#dump(tokenizer, open('tokenizer.pkl', 'wb'))

token_start = tokenizer.word_index[mark_start.strip()]
token_start

token_end = tokenizer.word_index[mark_end.strip()]
token_end

tokens_train = tokenizer.captions_to_tokens(captions_train_marked)

tokens_train[0]

captions_train_marked[0]

def get_random_caption_tokens(idx):
    result = []
    for i in idx:
        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]
        result.append(tokens)
    return result

def batch_generator(batch_size):
    while True:
        idx = np.random.randint(num_images_train, size=batch_size)
        transfer_values = transfer_values_train[idx]
        tokens = get_random_caption_tokens(idx)
        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)
        tokens_padded = pad_sequences(tokens, maxlen=max_tokens, padding='post', truncating='post')
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]
        x_data = {'decoder_input': decoder_input_data, 'transfer_values_input': transfer_values}
        y_data = {'decoder_output': decoder_output_data}
        yield (x_data, y_data)

batch_size = 128

generator = batch_generator(batch_size=batch_size)

batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]

batch_x['transfer_values_input'][0]

batch_x['decoder_input'][0]

batch_y['decoder_output'][0]

num_captions_train = [len(captions) for captions in captions_train]

total_num_captions_train = np.sum(num_captions_train)

steps_per_epoch = int(total_num_captions_train / batch_size)
steps_per_epoch

state_size = 512

embedding_size = 128

transfer_values_input = Input(shape=(transfer_values_size,), name='transfer_values_input')

decoder_transfer_map = Dense(state_size, activation='tanh', name='decoder_transfer_map')  #try using Maxout

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size, name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)

decoder_dense = Dense(num_words, activation='linear', name='decoder_output')

def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    return decoder_output

decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input], outputs=[decoder_output])

def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


optimizer = RMSprop(lr=1e-3)
tf.compat.v1.disable_eager_execution()
decoder_target = tf.compat.v1.placeholder(dtype='int32', shape=(None, None), name=None)

decoder_model.compile(optimizer=optimizer, loss=sparse_cross_entropy, target_tensors=[decoder_target])

path_checkpoint = '22_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, verbose=1, save_weights_only=True)

callback_tensorboard = TensorBoard(log_dir='./22_logs/', histogram_freq=0, write_graph=False)

callbacks = [callback_checkpoint, callback_tensorboard]

try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

decoder_model.fit_generator(generator=generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=1,
                            callbacks=callbacks)
#%%
from keras.models import model_from_json
# serialize model to JSON
model_json = decoder_model.to_json()
with open("decode_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("image_captioning.h5")
print("Saved model to disk")