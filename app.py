import gradio as gr
import os
from src.utils import list_models
from src.prediction import PredictorFatory

EXAMPLE_PATH = './assets/examples'
MODEL_PATH = './model/'
DEFAULT_MODEL = 'xception_299_299.hdf5'
EXAMPLES = [[EXAMPLE_PATH + '/' + file, DEFAULT_MODEL] for file in os.listdir(EXAMPLE_PATH)]
CLASSES = ['Figure 人物 ', 'Item 道具','Landscape 自然风光', 'Machine 机械', 'City 城市建筑', 'Indoor 室内']
article = """
<p style='text-align: center'>
    <a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a>
</p>
"""

def entry(image, model_choice):
  if 'xception' in model_choice:
    factory = PredictorFatory('xception', CLASSES, MODEL_PATH + model_choice)
    predictor = factory.get_predictor()
    return predictor.predict(image)
  elif model_choice=="SWIN2":
    pass

gr.Interface(fn=entry,  
            description=article,
            inputs=[gr.Image(), gr.inputs.Dropdown(choices=list_models(MODEL_PATH), type="value", default=DEFAULT_MODEL)],
            outputs=gr.Label(num_top_classes=len(CLASSES)),
            examples=EXAMPLES,
            title = "Anime Theme Classification").launch()
