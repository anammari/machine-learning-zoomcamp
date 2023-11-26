#!/usr/bin/env python
# coding: utf-8

import requests
from PIL import Image
from io import BytesIO
import tflite_runtime.interpreter as tflite
import numpy as np

target_size = (150, 150)

def get_image(url):
    # Send a HTTP request to the URL of the image
    response = requests.get(url)
    # Open the URL image as a PIL image object
    img = Image.open(BytesIO(response.content))
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    img = prepare_image(img, target_size) # Resize the image
    img_tensor = np.array(img, dtype='float32')
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.  # Normalize to [0,1]
    return img_tensor

def predict(url):
    img = get_image(url)
    img_tensor = preprocess_image(img)
    interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, img_tensor)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    wasp_prob = float_predictions[0]
    bee_prob = 1.0 - wasp_prob
    result = {'Bee': bee_prob, 'Wasp': wasp_prob}
    return result

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result