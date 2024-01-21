import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.text import Tokenizer

def load_image(path, size=None):
    img = Image.open(path)
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.array(img)
    img = img / 255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img

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
    
tokenizer = open('tokenizer.pkl','rb')
tokenizer = pickle.load(tokenizer)

# load json and create model
#json_file = open('decoder_model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)

# load weights into new model
#loaded_model.load_weights("image_captioning_model.h5")
loaded_model = tf.keras.models.load_model('model_and_weights.h5')
print("Loaded model from disk")

image_model = VGG16(include_top=True, weights='imagenet')

img_size = K.int_shape(image_model.input)[1:3]
img_size

#image_model.summary()

transfer_layer = image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)

img_size = K.int_shape(image_model.input)[1:3]
img_size

mark_start = 'ssss '
mark_end = ' eeee'
token_start = tokenizer.word_index[mark_start.strip()]
token_start


token_end = tokenizer.word_index[mark_end.strip()]
token_end

def generate_caption(image_path, max_tokens=30):
    image = load_image(image_path, size=img_size)
    image_batch = np.expand_dims(image, axis=0)
    transfer_values = image_model_transfer.predict(image_batch)
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {'transfer_values_input': transfer_values, 'decoder_input': decoder_input_data}
        decoder_output = loaded_model.predict(x_data)
        decoder_output
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        if token_int == token_end:
            break
        sampled_word = tokenizer.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
    output_tokens = decoder_input_data[0]
    output_tokens
    plt.imshow(image)
    plt.show()
    print("Predicted caption:")
    print(output_text)
    print()

#Generate an Image Caption
generate_caption("test1.jpg")

