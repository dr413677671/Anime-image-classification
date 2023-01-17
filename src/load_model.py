import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import onnxruntime
import numpy as np

def load_model_Xception(weight_path):
    base_model = tf.keras.applications.Xception(include_top=False, weights=None, classes=6,classifier_activation='softmax')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.layers[-1].output)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
    model = Model(inputs = base_model.inputs, outputs = outputs)
    model.load_weights(weight_path)
    return model
    # model.compile(optimizer='adam', metrics=['acc'],loss=tf.keras.losses.sparse_categorical_crossentropy)

def load_model_Matting(weight_path):
    sess = onnxruntime.InferenceSession(weight_path)
    return sess