from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.applications import VGG16
import joblib
from joblib import load, dump
from keras.preprocessing.text import Tokenizer
import base64
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)

sess = tf.compat.v1.Session()
global graph
graph = tf.compat.v1.get_default_graph()
with graph.as_default():
	set_session(sess)
	global model
	model = tf.compat.v1.keras.models.load_model('model_and_weights.h5')
	print("Loaded model from disk")

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

tokenizer = open('tokenizer.pkl', 'rb')
tokenizer = joblib.load(tokenizer)

image_model = VGG16(include_top=True, weights='imagenet')

img_size = K.int_shape(image_model.input)[1:3]
img_size

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


def load_image(path, size=None):
    img = Image.open(path)
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.array(img)
    img = img / 255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img

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
        decoder_output = model.predict(x_data)
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
    print("Predicted caption:")
    print(output_text)
    return output_text


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
	with graph.as_default():
		set_session(sess)
		image = request.args.get('x')
		image=image[23:]
		imgdata = base64.b64decode(image)
		fh = open("uploadimage.jpg", "wb")
		fh.write(imgdata)
		fh.close()
		print("fine2")
		description = generate_caption("uploadimage.jpg")
		print(description)
		return jsonify(description)
		return render_template('index.html', prediction = description)
	print("nahi")
    


if __name__ == '__main__':
	print("when")
	app.run(debug=False)
