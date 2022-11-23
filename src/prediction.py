from abc import abstractmethod, ABCMeta
import tensorflow as tf
import copy
from .preprocess import perprocess_infer
from .load_model import load_model_Xception


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(self):
        pass

class XceptionPredictor():
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

class PredictorFatory():
  def __init__(self, type, classes, model_path):
    self.type = type
    self.model_path = model_path
    self.classes = classes

  def get_predictor(self):
    if self.type == 'xception':
      model = load_model_Xception(self.model_path)
      return XceptionPredictor(model, copy.deepcopy(self.classes))