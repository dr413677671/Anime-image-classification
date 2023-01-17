from abc import abstractmethod, ABCMeta
import copy
from .preprocess import perprocess_infer, prep_matting, prep_seg
from .load_model import load_model_Xception, load_model_Matting
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.keras.utils import custom_object_scope

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(self):
        pass

class XceptionPredictor(Predictor):
    WIDTH = 299
    HEIGHT = 299
    def __init__(self, model, classes):
      self.model = model
      self.classes = classes
    def predict(self, inp) -> list:
      cls_count = len(self.classes)
      inp = perprocess_infer(inp, self.WIDTH, self.HEIGHT)
      prediction = self.model.predict(inp).flatten()
      confidences = {self.classes[i]: float(prediction[i]) for i in range(cls_count)}
      return confidences

class MatPredictor(Predictor):
  def __init__(self, sess):
    self.sess = sess
  def predict(self, src):
    src, bgr = prep_matting(src)
    pha, fgr = self.sess.run(['pha', 'fgr'], {'src': src, 'bgr': bgr})
    com = (pha * fgr + (1 - pha) * bgr)*255
    com = com.astype(np.uint8)
    fgr = (fgr*255).astype(np.uint8)
    cv2.imshow('fgr', fgr.transpose((0,2,3,1)).squeeze())
    cv2.imshow('com', com.transpose((0,2,3,1)).squeeze())
    cv2.waitKey()
    return pha, fgr 

class Segmentor(Predictor):
  def __init__(self, model):
    self.model = model
    self.models = dict()
  def predict(self, x):
    h, w, _ = x.shape
    x = prep_seg(x)
    y = self.model.predict(x)[0]
    y = cv2.resize(y, (w, h))
    _, y = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y = y.astype(np.uint8)
    # closed ops
    y = cv2.dilate(src=y, kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3)), iterations=4)
    y = cv2.erode(src=y, kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3)), iterations=2)
    
    # fill big hole
    y = FillHole(y)
    return  y

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    if len_contour == 0:
      return mask
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
 
    out = sum(contour_list)
    return out

class PredictorFatory():
  def __init__(self, classes, model_path):
    self.model_path = model_path
    self.classes = classes
    self.models = dict()

  def get_predictor(self, mode='xception'):
    
    if mode == 'xception':
      if 'xception' in self.models.keys():
        return self.models['xception']
      else:
        model = load_model_Xception(self.model_path)
        return XceptionPredictor(model, copy.deepcopy(self.classes))

  def get_matting(self, matting_weight):
      return self.get('matting', lambda: MatPredictor(load_model_Matting(matting_weight)))
  
  def get_seg(self, segmodel):
    if 'segmentation' not in self.models.keys():
      with tf.keras.utils.custom_object_scope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        self.models['segmentation'] = Segmentor(tf.keras.models.load_model(segmodel))
    return self.models['segmentation']

  def get(self, name, func):
    if name in self.models.keys():
      return self.models[name]
    else:
      self.models[name] = func()
      return self.models[name]