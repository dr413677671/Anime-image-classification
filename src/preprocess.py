import tensorflow as tf

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