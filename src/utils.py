import os

def list_models(path):
    classes = []
    for cls in os.listdir(path):
        if os.path.isfile(path +'/' + cls):
            classes.append(cls)
    return classes