import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json

class_names = {}
image_size = 224


#VARS
def process_image(image):
    image = np.squeeze(image)
    image = tf.image.resize(image, (image_size, image_size))/255.0

    return image


def predict(image_path, model, top_k_nums='3', mapping=None):
    image = Image.open(image_path)
    curr_image = np.asarray(image)
    processed = process_image(curr_image)
    #predict image
    prediction = model.predict(np.expand_dims(processed, axis=0))
    top_k_vals, top_k_indices = tf.math.top_k(prediction, int(top_k_nums))
    if mapping is not None:
        with open(mapping, 'r') as f:
            class_names = json.load(f)

        top_k_labels = [class_names[str(index+1)]
                        for index in top_k_indices[0].numpy()]
        return top_k_vals.numpy()[0], top_k_labels
    else:
        return top_k_vals.numpy()[0], top_k_indices.numpy()[0]


#ARGPARSE
parser = argparse.ArgumentParser(
    description='What flower would you like to predict?')
parser.add_argument('arg1')
parser.add_argument('arg2')
parser.add_argument('--top_k')
parser.add_argument('--category_names')

args = parser.parse_args()
path_to_image = args.arg1
saved_keras_model_filepath = args.arg2

reloaded_keras_model = tf.keras.models.load_model(
    saved_keras_model_filepath, custom_objects={'KerasLayer': hub.KerasLayer})
top_k = args.top_k
map_labels = args.category_names

if top_k is None:
    top_k = 3

probs, classes = predict(
    path_to_image, reloaded_keras_model, top_k, map_labels)

print('Probabilities:', probs, 'Classes:', classes)
