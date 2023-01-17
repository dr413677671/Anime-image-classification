import tensorflow as tf
import numpy as np
import cv2

def perprocess_train_no_aug(image_name, WIDTH, HEIGHT):
    image_s = tf.io.read_file(image_name)
    raw_image = tf.image.decode_image(image_s,channels=3)
    raw_image.set_shape([None,None,None])
    image = tf.image.convert_image_dtype(raw_image, tf.float32)
    image = tf.image.resize(image, (WIDTH, HEIGHT))
    return image, raw_image

# @tf.function   
def perprocess_infer(image_s, WIDTH, HEIGHT):
    # image_s = tf.io.read_file(image_name)
    image = tf.image.convert_image_dtype(image_s, tf.float32)
    # image = tf.image.decode_image(image,channels=3)
    # image.set_shape([None,None,None])
    image = tf.image.resize(image, (WIDTH, HEIGHT))
    image = tf.reshape(image, shape=[1, image.get_shape()[0], image.get_shape()[1], image.get_shape()[2]])
    return image

def prep_matting(src):
    HEIGHT = 1080
    WIDTH = 1920
    src = np.expand_dims(cv2.resize(src, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC).transpose((2,0,1)), axis=0).astype(np.float32)
    src /= 255.
    # src = np.random.normal(size=(1, 3, HEIGHT, WIDTH)).astype(np.float32)
    bgr = np.ones((1, 3, HEIGHT, WIDTH)).astype(np.float32)
    return src, bgr

def prep_seg(image):
    H, W = 512, 512
    x = cv2.resize(image, (W, H))
    x = x.astype(np.float32)
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    return x