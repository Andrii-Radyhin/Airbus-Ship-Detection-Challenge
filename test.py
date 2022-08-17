from config import *
from utils import generators

BASE_DIR = ''
TEST_DIR = BASE_DIR + 'test_v2/'

test_imgs = ['00dc34840.jpg', '00c3db267.jpg', '00aa79c47.jpg', '00a3a9d72.jpg']

import tensorflow as tf
import cv2
from tensorflow import keras
from matplotlib import pyplot as plt
import os
import numpy as np

def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(test_dir,img)
    img = cv2.imread(rgb_path)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred

fullres_model = keras.models.load_model('fullres_model & weights/fullres_model.h5')

rows = 1
columns = 2
for i in range(len(test_imgs)):
    img, pred = generators.gen_pred(TEST_DIR, test_imgs[i], fullres_model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
