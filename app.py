import gradio as gr
import os
import numpy as np
from src.utils import list_models
from src.prediction import PredictorFatory
import cv2
import copy

EXAMPLE_PATH = './assets/examples'
MODEL_PATH = './model/'
DEFAULT_MODEL = 'xception_299_299.hdf5'
EXAMPLES = [[EXAMPLE_PATH + '/' + file, DEFAULT_MODEL, True] for file in os.listdir(EXAMPLE_PATH)]
CLASSES = ['Figure 人物 ', 'Item 道具','Landscape 自然风光', 'Machine 机械', 'City 城市建筑', 'Indoor 室内']
LABELS = ['Figure', 'Item','Landscape', 'Machine', 'City', 'Indoor']
MATTING_WEIGHT = './seg/onnx_mobilenetv2_hd.onnx'
SEG_MODEL = './seg/seg.h5'

article = """
<p style='text-align: center'>
    <a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a>
</p>
"""

def entry(image, model_choice, enhance=False):
  factory = PredictorFatory(CLASSES, MODEL_PATH + model_choice)
  cls_ret = None
  if 'xception' in model_choice:
    predictor = factory.get_predictor('xception')
    
    
    if 'matting' in model_choice:
      mat_pridictior = factory.get_matting(MATTING_WEIGHT)
      pha, fgr  = mat_pridictior.predict(image)
      cls_ret = predictor.predict(image)
    elif enhance:
      segmentor = factory.get_seg(SEG_MODEL)
      seg = segmentor.predict(copy.deepcopy(image))

      inv_mask = seg.astype(bool)
      inv_mask = ~inv_mask
      inv_mask = inv_mask.astype(np.uint8)
      cls_image = copy.deepcopy(image)*np.expand_dims(inv_mask, axis=-1)
      cls_ret = predictor.predict(cls_image)

      # max connected components
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=8)
      area = 0
      weight = 0.9
      for i in range(num_labels):
        if stats[i][-1] > area:
          area = stats[i][-1]
          idx = i
      if area > image.shape[0] * image.shape[1] * 0.02:
        cls_ret[CLASSES[0]] = weight + (1-weight) * cls_ret[CLASSES[0]]
        cv2.putText(image, LABELS[0], (int(centroids[idx][0]), int(centroids[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)

      contours, hierarchy = cv2.findContours(seg.astype(np.uint8), cv2.RETR_LIST, 2)
      cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
      cv2.putText(image, LABELS[np.where(list(cls_ret.values())==np.max(list(cls_ret.values())[1:]))[0][0]], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)

      # cv2.imshow("seg", seg)
      # cv2.imshow("cls_image", cls_image)
      # cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
      
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

    return cls_ret, image

  elif "SWIN2" in model_choice:
    pass

gr.Interface(fn=entry,  
            description=article,
            inputs=[gr.Image(), gr.inputs.Dropdown(choices=list_models(MODEL_PATH), type="value", default=DEFAULT_MODEL), gr.inputs.Checkbox(default=False)],
            outputs=[gr.Label(num_top_classes=len(CLASSES)), gr.Image()],
            examples=EXAMPLES,
            title = "Anime Theme Classification").launch()
